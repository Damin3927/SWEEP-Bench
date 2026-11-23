from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import math
import os
import select
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.mesh_import import (
    GsoManifest,
    ImportedMeshEntry,
    default_cache_dir,
    load_manifest,
    maniskill_asset_dir,
    write_json_atomic,
)
from sweep_bench.assets.procedural import PROCEDURAL_SPECS
from sweep_bench.code_provenance import home_relative
from sweep_bench.tasks.conveyor_base.config import ConveyorMotionRandomization
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

    from sweep_bench.assets.catalog import ObjectSpec

logger = logging.getLogger(__name__)

Source = Literal["procedural", "gso", "ycb", "robocasa", "polyhaven"]
MESH_SOURCES: tuple[Source, ...] = ("gso", "ycb", "robocasa", "polyhaven")
CollisionMode = Literal["primitive", "coacd", "convex_hull"]
Verdict = Literal["pass", "reject"]


VET_SPEED_DEFAULT = 0.15
_DEFAULT_VET_ROBOT = "panda"

_GRIPPER_OPENING = 0.08
_GRASP_MARGIN = 0.005
_MAX_RIDER_FOOTPRINT = 0.07
_BELT_HALF_WIDTH = 0.08

_TIP_DEG_MAX = 45.0
_SINK_TOL = 0.03
_LAUNCH_TOL = 0.06
_RAIL_MARGIN = 0.005
_MIN_TRACK_FRAC = 0.4
_EDGE_MARGIN = 0.06

_WINDOW_STEPS = 40
_MIN_ONBELT_STEPS = 10
_SETTLE_STEPS = 25
_GRACE_STEPS = 5

_CONTROL_DT = 1.0 / 20.0

MANIFEST_SCHEMA_VERSION = 1

VETTING_ALGO_VERSION = 2


@dataclass(frozen=True)
class VetCandidate:
    id: str
    source: Source
    role: str
    collision_mode: CollisionMode
    builder_fn: Callable[[ManiSkillScene], ActorBuilder]
    patch_radius: float | None
    grasp_width: float | None
    bbox: tuple[float, float, float]
    visual_rel: str | None = None
    collision_rel: str | None = None
    geometry: str | None = None

    physics: tuple[object, ...] | None = None

    _RIDER_ROLES = ("graspable_rider", "container", "insert_socket")

    @property
    def is_rider(self) -> bool:
        return self.role in self._RIDER_ROLES

    @property
    def belt_patch_radius(self) -> float | None:
        return self.patch_radius if self.is_rider else None


@functools.cache
def _procedural_by_id() -> dict[str, ObjectSpec]:
    return {spec.id: spec for spec in PROCEDURAL_SPECS}


def procedural_candidates() -> list[VetCandidate]:
    return [
        VetCandidate(
            id=spec.id,
            source="procedural",
            role=spec.role,
            collision_mode="primitive",
            builder_fn=spec.builder_fn,
            patch_radius=spec.patch_radius,
            grasp_width=spec.grasp_width,
            bbox=spec.bbox,
            geometry=spec_geometry_stamp(spec),
        )
        for spec in PROCEDURAL_SPECS
    ]


_GEOMETRY_STAMP_VERSION = "g2"


@functools.lru_cache(maxsize=4096)
def _file_digest(path: str) -> str:
    file = Path(path)
    if not file.is_file():
        return "missing"
    digest = hashlib.sha256()
    with file.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def entry_geometry_stamp(entry: ImportedMeshEntry) -> str:
    physics = ":".join(
        [
            _number(entry.density),
            _number(entry.friction),
            ",".join(f"{v:.9g}" for v in entry.native_center),
            str(entry.collision_parts),
        ]
    )
    return _identity_stamp(
        source=entry.source,
        role=entry.role,
        collision_mode=entry.collision_method,
        bbox=entry.bbox,
        patch_radius=entry.patch_radius,
        grasp_width=entry.grasp_width,
        content=geometry_stamp(entry.scale, entry.bbox, entry.visual_file, entry.collision_file) + "|" + physics,
    )


def geometry_stamp(
    scale: float,
    bbox: tuple[float, float, float],
    visual_file: str | None = None,
    collision_file: str | None = None,
) -> str:
    parts = [_GEOMETRY_STAMP_VERSION, f"{scale:.9g}", ",".join(f"{v:.9g}" for v in bbox)]
    for file in (visual_file, collision_file):
        if file is not None:
            parts.append(_file_digest(file))
    return ":".join(parts)


def _number(value: float | None) -> str:
    return "none" if value is None else f"{value:.9g}"


def _stamp_value(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, (int, float)):
        return f"{value:.9g}"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (tuple, list)):
        return "(" + ",".join(_stamp_value(v) for v in value) + ")"
    raise TypeError(
        f"cannot stamp builder argument of type {type(value).__name__!r} deterministically; "
        "add a stable rendering for it in vetting._stamp_value"
    )


@functools.lru_cache(maxsize=256)
def _implementation_digest(fn: Callable[..., object]) -> str:
    import inspect

    try:
        source_file = inspect.getsourcefile(fn)
        source = Path(source_file).read_text(encoding="utf-8") if source_file else inspect.getsource(fn)
    except (OSError, TypeError):
        return "no-source"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _builder_descriptor(builder_fn: Callable[..., object]) -> str:
    args: list[str] = []
    fn: Callable[..., object] = builder_fn
    while isinstance(fn, functools.partial):
        args.extend(_stamp_value(a) for a in fn.args)
        args.extend(f"{k}={_stamp_value(v)}" for k, v in sorted(fn.keywords.items()))
        fn = fn.func
    name = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', type(fn).__name__)}"
    return f"{name}@{_implementation_digest(fn)}(" + ",".join(args) + ")"


def spec_geometry_stamp(spec: ObjectSpec) -> str:
    content = ":".join(
        [
            "proc",
            _number(spec.mass),
            hashlib.sha256(_builder_descriptor(spec.builder_fn).encode()).hexdigest()[:16],
        ]
    )
    return _identity_stamp(
        source=spec.source,
        role=spec.role,
        collision_mode="primitive",
        bbox=spec.bbox,
        patch_radius=spec.patch_radius,
        grasp_width=spec.grasp_width,
        content=content,
    )


def imported_mesh_candidates(manifest: GsoManifest, cache_dir: Path) -> list[VetCandidate]:
    candidates: list[VetCandidate] = []
    for entry in manifest.entries:
        spec = entry.to_spec()
        candidates.append(
            VetCandidate(
                id=entry.id,
                source=_mesh_source(entry.source),
                role=entry.role,
                collision_mode=entry.collision_method,
                builder_fn=spec.builder_fn,
                patch_radius=entry.patch_radius,
                grasp_width=entry.grasp_width,
                bbox=entry.bbox,
                visual_rel=_relpath(entry.visual_file, cache_dir),
                collision_rel=_relpath(entry.collision_file, cache_dir),
                geometry=entry_geometry_stamp(entry),
                physics=(
                    entry.scale,
                    entry.native_center,
                    entry.density,
                    entry.friction,
                    entry.collision_parts,
                ),
            )
        )
    return candidates


def _mesh_or_procedural_source(source: str) -> Source:
    if source == "procedural":
        return "procedural"
    return _mesh_source(source)


def _mesh_source(source: str) -> Source:
    if source == "gso":
        return "gso"
    if source == "robocasa":
        return "robocasa"
    if source == "polyhaven":
        return "polyhaven"
    if source == "ycb":
        return "ycb"
    raise ValueError(f"manifest entry has unknown source {source!r} (expected one of {MESH_SOURCES})")


def _relpath(path: str, cache_dir: Path) -> str:
    resolved = Path(path).resolve()
    for root in (cache_dir, maniskill_asset_dir()):
        if resolved.is_relative_to(root.resolve()):
            return str(resolved.relative_to(root.resolve()))
    return str(resolved)


def load_candidates(
    sources: Sequence[Source],
    *,
    gso_manifest_path: str | Path | None = None,
    cache_dir: Path | None = None,
    sample_limit: int | None = None,
    only: Sequence[str] | None = None,
) -> list[VetCandidate]:
    cache = cache_dir or default_cache_dir()
    cap = None if only is not None else sample_limit
    out: list[VetCandidate] = []
    if "procedural" in sources:
        procedural = procedural_candidates()
        out.extend(procedural if cap is None else procedural[:cap])
    wanted_mesh = [s for s in MESH_SOURCES if s in sources]
    if wanted_mesh:
        manifest_path = Path(gso_manifest_path) if gso_manifest_path else (cache / "gso_manifest.json")
        if manifest_path.is_file():
            everything = imported_mesh_candidates(load_manifest(manifest_path), cache)
            for mesh_source in wanted_mesh:
                same = [c for c in everything if c.source == mesh_source]
                out.extend(same if cap is None else same[:cap])
        else:
            logger.warning("imported-mesh manifest not found at %s; skipping %s candidates", manifest_path, wanted_mesh)
    if only is not None:
        by_id = {c.id: c for c in out}
        missing = [i for i in only if i not in by_id]
        if missing:
            logger.warning("requested --only ids not found: %s", missing)
        out = [by_id[i] for i in only if i in by_id]
    return out


def candidate_fingerprint(candidate: VetCandidate) -> str:
    if candidate.geometry is not None:
        return candidate.geometry
    return _identity_stamp(
        source=candidate.source,
        role=candidate.role,
        collision_mode=candidate.collision_mode,
        bbox=candidate.bbox,
        patch_radius=candidate.patch_radius,
        grasp_width=candidate.grasp_width,
        content=None if candidate.physics is None else "physics:" + repr(candidate.physics),
    )


def _identity_stamp(
    *,
    source: str,
    role: str,
    collision_mode: str,
    bbox: tuple[float, float, float],
    patch_radius: float | None,
    grasp_width: float | None,
    content: str | None,
) -> str:
    payload = "|".join(
        [
            _GEOMETRY_STAMP_VERSION,
            source,
            role,
            collision_mode,
            ",".join(f"{v:.9g}" for v in bbox),
            _number(patch_radius),
            _number(grasp_width),
            content or "no-content",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class VetResult:
    id: str
    source: Source
    role: str
    collision_mode: CollisionMode
    verdict: Verdict
    reason: str
    checks: dict[str, bool | None]
    metrics: dict[str, float | bool]
    visual_rel: str | None = None
    collision_rel: str | None = None
    geometry: str | None = None

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "id": self.id,
            "source": self.source,
            "role": self.role,
            "collision_mode": self.collision_mode,
            "verdict": self.verdict,
            "reason": self.reason,
            "checks": self.checks,
            "metrics": self.metrics,
        }
        if self.visual_rel is not None:
            out["visual_file_rel"] = self.visual_rel
        if self.collision_rel is not None:
            out["collision_file_rel"] = self.collision_rel
        if self.geometry is not None:
            out["geometry"] = self.geometry
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> VetResult:
        def _s(key: str) -> str:
            return str(data[key])

        raw_verdict = _s("verdict")
        if raw_verdict == "pass":
            verdict: Verdict = "pass"
        elif raw_verdict == "reject":
            verdict = "reject"
        else:
            raise ValueError(f"bad result verdict {raw_verdict!r}")

        raw_mode = _s("collision_mode")
        if raw_mode == "primitive":
            mode: CollisionMode = "primitive"
        elif raw_mode == "coacd":
            mode = "coacd"
        elif raw_mode == "convex_hull":
            mode = "convex_hull"
        else:
            raise ValueError(f"bad collision_mode {raw_mode!r}")

        raw_source = _s("source")
        source = _mesh_or_procedural_source(raw_source)

        checks = data.get("checks")
        metrics = data.get("metrics")
        if not isinstance(checks, dict) or not isinstance(metrics, dict):
            raise ValueError(f"malformed result payload for {data.get('id')!r}")
        return cls(
            id=_s("id"),
            source=source,
            role=_s("role"),
            collision_mode=mode,
            verdict=verdict,
            reason=_s("reason"),
            checks={str(k): (None if v is None else bool(v)) for k, v in checks.items()},
            metrics={str(k): v for k, v in metrics.items() if isinstance(v, (int, float, bool))},
            visual_rel=None if "visual_file_rel" not in data else str(data["visual_file_rel"]),
            collision_rel=None if "collision_file_rel" not in data else str(data["collision_file_rel"]),
            geometry=None if data.get("geometry") is None else str(data["geometry"]),
        )


def stale_verdict_reason(result: VetResult, entry: ImportedMeshEntry | None) -> str | None:
    if entry is None:
        if result.source in MESH_SOURCES:
            return f"{result.source} row has no entry in the imported-mesh manifest"
        spec = _procedural_by_id().get(result.id)
        if spec is None:
            return "procedural row names no current spec, so the object it measured no longer exists"
        return _stale_reason(result, role=spec.role, geometry=spec_geometry_stamp(spec))
    return _stale_reason(result, role=entry.role, geometry=entry_geometry_stamp(entry))


def candidate_stale_reason(result: VetResult, candidate: VetCandidate) -> str | None:
    return _stale_reason(result, role=candidate.role, geometry=candidate.geometry)


def _stale_reason(result: VetResult, *, role: str, geometry: str | None) -> str | None:
    if result.role != role:
        return f"role {result.role!r} != current {role!r}"
    if result.geometry is None:
        return "row records no geometry, so it cannot show what it measured"
    if geometry is None:
        return "the current object carries no geometry stamp, so the row cannot be shown to match it"
    if result.geometry != geometry:
        return f"geometry {result.geometry} != current {geometry}"
    return None


class _VetEnv(ConveyorWorkspaceEnv):
    default_prompt = "vet object"
    RandomizationConfig = ConveyorMotionRandomization
    randomization: ConveyorMotionRandomization

    def __init__(
        self,
        *args,
        vet_builder: Callable[[ManiSkillScene], ActorBuilder],
        vet_patch: float | None,
        vet_bbox: tuple[float, float, float],
        **kwargs,
    ) -> None:
        self._vet_builder = vet_builder
        self._vet_patch = vet_patch
        self._vet_bbox = vet_bbox
        self.randomization = ConveyorMotionRandomization(speed=False, direction=False, spawn_phase_range=0.0)
        super().__init__(*args, **kwargs)

    def set_candidate(
        self,
        builder_fn: Callable[[ManiSkillScene], ActorBuilder],
        patch_radius: float | None,
        bbox: tuple[float, float, float],
    ) -> None:
        self._vet_builder = builder_fn
        self._vet_patch = patch_radius
        self._vet_bbox = bbox

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        builder = self._vet_builder(self.scene)
        builder.initial_pose = sapien.Pose(p=[ox, oy, oz + self._conveyor_deck_height + 0.05])
        self.obj = builder.build(name="vet_obj")
        self.obj.angular_damping = 5.0
        if self._vet_patch is not None:
            self.belt.register_rider(self.obj, patch_radius=self._vet_patch)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        ox, oy, oz = self.workspace_offset
        device = self.device
        rest_z = oz + self._conveyor_deck_height + self._vet_bbox[2] + 0.001
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self._vet_patch is not None:
            pos[:, 1] = oy - self._stable_half() * self.belt.direction_sign[env_idx]
        else:
            pos[:, 1] = oy
        pos[:, 2] = rest_z
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        self.obj.set_pose(Pose.create_from_pq(pos, q))
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        self.obj.set_linear_velocity(vel)
        self.obj.set_angular_velocity(torch.zeros((b, 3), device=device))

    def _stable_half(self) -> float:
        footprint = max(self._vet_bbox[0], self._vet_bbox[1])
        return max(self._conveyor_half_length - footprint - _EDGE_MARGIN - 0.02, 0.0)

    def evaluate(self):
        return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)


def make_vet_env(first: VetCandidate, *, robot_uids: str = "panda") -> _VetEnv:
    return _VetEnv(
        robot_uids=robot_uids,
        obs_mode="state",
        control_mode="pd_joint_pos",
        scene_style="table",
        num_envs=1,
        vet_builder=first.builder_fn,
        vet_patch=first.belt_patch_radius,
        vet_bbox=first.bbox,
    )


def _tilt_deg(quat: torch.Tensor) -> float:
    rot = quaternion_to_matrix(quat)
    cos = float(rot[0, 2, 2].clamp(-1.0, 1.0).item())
    return math.degrees(math.acos(cos))


@dataclass
class _RideTrace:
    finite: bool = True
    max_tilt_deg: float = 0.0
    max_x_drift: float = 0.0
    min_z: float = math.inf
    max_z: float = -math.inf
    y_start: float = 0.0
    y_last_onbelt: float = 0.0
    onbelt_steps: int = 0
    reached_edge: bool = False
    reasons: list[str] = field(default_factory=list)


def _run_sim(env: _VetEnv, candidate: VetCandidate, *, speed: float, seed: int) -> _RideTrace:
    env.set_candidate(candidate.builder_fn, candidate.belt_patch_radius, candidate.bbox)
    env.conveyor_speed = speed if candidate.is_rider else 0.0
    env.reset(seed=seed, options={"reconfigure": True})

    ox, oy, _oz = env.workspace_offset
    footprint = max(candidate.bbox[0], candidate.bbox[1])
    stable_half = env._conveyor_half_length - footprint - _EDGE_MARGIN
    rail_limit = _BELT_HALF_WIDTH - footprint - _RAIL_MARGIN

    trace = _RideTrace()
    trace.y_start = float(env.obj.pose.p[0, 1].item())
    trace.y_last_onbelt = trace.y_start

    hold = env.agent.robot.get_qpos()[:, :8].clone()
    steps = _WINDOW_STEPS if candidate.is_rider else _SETTLE_STEPS
    for step in range(steps):
        p = env.obj.pose.p[0]
        x, y, z = float(p[0].item()), float(p[1].item()), float(p[2].item())
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            trace.finite = False
            break
        tilt = _tilt_deg(env.obj.pose.q)
        trace.min_z = min(trace.min_z, z)
        trace.max_z = max(trace.max_z, z)
        trace.max_x_drift = max(trace.max_x_drift, abs(x - ox))
        settled = step >= _GRACE_STEPS
        if settled:
            trace.max_tilt_deg = max(trace.max_tilt_deg, tilt)

        if candidate.is_rider:
            if abs(y - oy) < stable_half:
                trace.onbelt_steps += 1
                trace.y_last_onbelt = y
                if settled and rail_limit > 0 and abs(x - ox) > rail_limit:
                    trace.reasons.append(
                        f"drifted into a belt rail (|x-center|={abs(x - ox):.3f} m > {rail_limit:.3f})"
                    )
                    break
                if settled and tilt > _TIP_DEG_MAX:
                    trace.reasons.append(f"tipped over (tilt {tilt:.0f} deg > {_TIP_DEG_MAX:.0f})")
                    break
            else:
                trace.reached_edge = True
                break
        with torch.no_grad():
            env.step(hold)

    return trace


def vet_candidate(env: _VetEnv, candidate: VetCandidate, *, speed: float, seed: int) -> VetResult:
    checks: dict[str, bool | None] = {"grasp_fit": None, "rails_clear": None, "settle": None, "belt_stable": None}
    reasons: list[str] = []

    footprint = max(candidate.bbox[0], candidate.bbox[1])
    deck_z = env.workspace_offset[2] + env._conveyor_deck_height

    if candidate.role == "graspable_rider":
        gw = candidate.grasp_width
        ok = gw is not None and gw < _GRIPPER_OPENING - _GRASP_MARGIN
        checks["grasp_fit"] = ok
        if not ok:
            reasons.append(f"grasp_width {gw} does not fit the gripper (< {_GRIPPER_OPENING - _GRASP_MARGIN:.3f} m)")

    if candidate.is_rider:
        ok = footprint <= _MAX_RIDER_FOOTPRINT + 1e-9
        checks["rails_clear"] = ok
        if not ok:
            reasons.append(
                f"footprint half-extent {footprint:.3f} m exceeds the belt half-width {_MAX_RIDER_FOOTPRINT} m"
            )

    try:
        trace = _run_sim(env, candidate, speed=speed, seed=seed)
    except Exception as e:  # noqa: BLE001 — a bad mesh/builder must reject, not abort the batch
        logger.warning("vet sim failed to build/step %s: %s", candidate.id, e)
        return VetResult(
            id=candidate.id,
            source=candidate.source,
            role=candidate.role,
            collision_mode=candidate.collision_mode,
            verdict="reject",
            reason=f"sim build/step failed: {e}",
            checks=checks,
            metrics={"finite": False},
            visual_rel=candidate.visual_rel,
            collision_rel=candidate.collision_rel,
            geometry=candidate_fingerprint(candidate),
        )

    expected_advance = speed * trace.onbelt_steps * _CONTROL_DT if candidate.is_rider else 0.0
    y_advance = trace.y_last_onbelt - trace.y_start
    metrics: dict[str, float | bool] = {
        "max_tilt_deg": round(trace.max_tilt_deg, 2),
        "max_x_drift": round(trace.max_x_drift, 4),
        "min_z": round(trace.min_z, 4),
        "max_z": round(trace.max_z, 4),
        "y_advance": round(y_advance, 4),
        "expected_advance": round(expected_advance, 4),
        "finite": trace.finite,
    }

    top_ceiling = deck_z + candidate.bbox[2] * 2 + _LAUNCH_TOL
    settle_ok = trace.finite and trace.min_z > deck_z - _SINK_TOL and trace.max_z < top_ceiling
    checks["settle"] = settle_ok
    if not trace.finite:
        reasons.append("collision exploded (pose went non-finite)")
    elif trace.min_z <= deck_z - _SINK_TOL:
        reasons.append(f"sank through the belt (min z {trace.min_z:.3f} < deck {deck_z:.3f})")
    elif trace.max_z >= top_ceiling:
        reasons.append(f"launched off the belt (max z {trace.max_z:.3f})")

    if candidate.is_rider:
        reasons.extend(trace.reasons)
        stable = settle_ok and not trace.reasons
        if stable and trace.onbelt_steps < _MIN_ONBELT_STEPS and not trace.reached_edge:
            stable = False
            reasons.append(f"left the belt almost immediately ({trace.onbelt_steps} on-belt steps)")
        if stable and y_advance < _MIN_TRACK_FRAC * expected_advance:
            stable = False
            reasons.append(f"did not ride the belt (advanced {y_advance:.3f} m vs expected {expected_advance:.3f})")
        checks["belt_stable"] = stable

    applicable = [v for v in checks.values() if v is not None]
    verdict: Verdict = "pass" if all(applicable) else "reject"
    reason = "" if verdict == "pass" else "; ".join(reasons) or "failed a vet check"
    return VetResult(
        id=candidate.id,
        source=candidate.source,
        role=candidate.role,
        collision_mode=candidate.collision_mode,
        verdict=verdict,
        reason=reason,
        checks=checks,
        metrics=metrics,
        visual_rel=candidate.visual_rel,
        collision_rel=candidate.collision_rel,
        geometry=candidate_fingerprint(candidate),
    )


def run_vetting(
    candidates: Sequence[VetCandidate],
    *,
    speed: float = VET_SPEED_DEFAULT,
    seed: int = 0,
    robot_uids: str = "panda",
    on_result: Callable[[VetResult], None] | None = None,
) -> list[VetResult]:
    if not candidates:
        return []
    env = make_vet_env(candidates[0], robot_uids=robot_uids)
    results: list[VetResult] = []
    try:
        for candidate in candidates:
            result = vet_candidate(env, candidate, speed=speed, seed=seed)
            results.append(result)
            if on_result is not None:
                on_result(result)
            logger.info(
                "%-8s %-40s %s%s",
                result.verdict.upper(),
                result.id,
                result.collision_mode,
                "" if result.verdict == "pass" else f"  <- {result.reason}",
            )
    finally:
        env.close()
    return results


_WORKER_STARTUP_TIMEOUT = 150.0
_WORKER_OBJECT_TIMEOUT = 60.0
_READY_TAG = "@@READY@@"
_RESULT_TAG = "@@RESULT@@"

MAX_ZERO_PROGRESS_CANDIDATES = 3


class ZeroProgressAbort(RuntimeError):
    pass


def _crash_result(candidate: VetCandidate, key: str, detail: str) -> VetResult:
    return VetResult(
        id=candidate.id,
        source=candidate.source,
        role=candidate.role,
        collision_mode=candidate.collision_mode,
        verdict="reject",
        reason=f"{key} ({detail})",
        checks={"grasp_fit": None, "rails_clear": None, "settle": None, "belt_stable": None},
        metrics={"finite": False},
        visual_rel=candidate.visual_rel,
        collision_rel=candidate.collision_rel,
        geometry=candidate_fingerprint(candidate),
    )


@dataclass
class _WorkerOutcome:
    results: list[VetResult]
    status: Literal["done", "crash", "timeout"]
    detail: str
    stderr_tail: list[str] = field(default_factory=list)


def _resume_vet(
    candidates: Sequence[VetCandidate],
    run_worker: Callable[[Sequence[VetCandidate], Callable[[VetResult], None] | None], _WorkerOutcome],
    on_result: Callable[[VetResult], None] | None = None,
    *,
    max_zero_progress: int = MAX_ZERO_PROGRESS_CANDIDATES,
) -> list[VetResult]:
    by_id: dict[str, VetResult] = {}

    def record(result: VetResult) -> None:
        by_id[result.id] = result
        if on_result is not None:
            on_result(result)
        logger.warning("REJECT   %-40s %s", result.id, result.reason)

    held: list[VetCandidate] = []
    deciding: set[str] = set()
    zero_progress = 0
    tried_by_id: dict[str, list[_WorkerOutcome]] = {}
    pending: list[VetCandidate] = list(candidates)
    outcome = _WorkerOutcome([], "done", "no worker ran")
    while pending:
        slice_ = pending
        outcome = run_worker(slice_, on_result)
        for r in outcome.results:
            by_id[r.id] = r
        done = len(outcome.results)
        if done == len(slice_):
            if outcome.status != "done":
                logger.warning("worker exited badly after reporting every result: %s", outcome.detail)
            pending = []
        else:
            culprit = slice_[done]
            tried_by_id.setdefault(culprit.id, []).append(outcome)
            pending = list(slice_[done + 1 :])
            if done > 0 or culprit.id in deciding:
                deciding.discard(culprit.id)
                if done > 0:
                    zero_progress = 0
                record(
                    _crash_result(
                        culprit,
                        "timeout" if outcome.status == "timeout" else "sim_crash",
                        _attempt_detail(tried_by_id[culprit.id]),
                    )
                )
            else:
                held.append(culprit)
                zero_progress += 1
                logger.warning("HOLD     %-40s worker reported nothing (%s)", culprit.id, outcome.detail)
                if zero_progress >= max_zero_progress:
                    raise ZeroProgressAbort(_zero_progress_message(held, outcome, max_zero_progress))
        if done > 0 and held:
            promoted = held.pop(0)
            deciding.add(promoted.id)
            pending = [promoted, *pending]
    if held:
        raise ZeroProgressAbort(_zero_progress_message(held, outcome, max_zero_progress, exhausted=True))
    return [by_id[c.id] for c in candidates]


def _attempt_detail(tried: Sequence[_WorkerOutcome]) -> str:
    if len(tried) == 1:
        only = tried[0]
        return only.detail if not only.stderr_tail else f"{only.detail}; stderr: {_stderr_summary(only.stderr_tail)}"
    parts = []
    for attempt, outcome in enumerate(tried, start=1):
        part = f"#{attempt} {outcome.detail}"
        if outcome.stderr_tail:
            part += f"; stderr: {_stderr_summary(outcome.stderr_tail, max_lines=2, max_chars=200)}"
        parts.append(part)
    return f"{len(tried)} attempts; " + "; ".join(parts)


def _zero_progress_message(
    blamed: Sequence[VetCandidate], outcome: _WorkerOutcome, limit: int, *, exhausted: bool = False
) -> str:
    n = len(blamed)
    lines = [
        f"the candidate list ran out while {n} candidate(s) were still waiting for a deciding attempt."
        if exhausted
        else f"{limit} candidates in a row reported no verdict, with no worker vetting anything in between.",
        "Aborting the run: a worker that vets nothing is evidence about the HARNESS, not about the "
        "objects (no free GPU, a CUDA OOM, a renderer that cannot initialize, a --manifest whose ids "
        "the worker cannot resolve). Blaming the first unreported candidate each time would write one "
        "permanent reject per candidate, which --resume never revisits.",
        f"Nothing was recorded for: {', '.join(r.id for r in blamed)}",
        f"Last worker: {outcome.status} ({outcome.detail})",
    ]
    if outcome.stderr_tail:
        lines.append("Worker stderr (tail):")
        lines.extend(f"  {line}" for line in outcome.stderr_tail)
    else:
        lines.append("Worker stderr: empty (the worker died before writing anything).")
    if exhausted:
        lines.append(
            "No later worker could show whether the harness works, so these stay unvetted. To record a "
            "verdict for one, vet it in a list with objects known to pass -- a worker that gets through "
            "those and then dies is evidence about the object."
        )
    else:
        lines.append(
            f"Fix the cause and re-run -- those candidates are still unvetted. If they really are {limit} "
            f"bad assets in a row, re-run with --max-zero-progress {limit + 1}."
        )
    return "\n".join(lines)


def run_isolated(
    candidates: Sequence[VetCandidate],
    *,
    speed: float = VET_SPEED_DEFAULT,
    seed: int = 0,
    robot_uids: str = "panda",
    startup_timeout: float = _WORKER_STARTUP_TIMEOUT,
    object_timeout: float = _WORKER_OBJECT_TIMEOUT,
    gso_manifest_path: str | Path | None = None,
    cache_dir: Path | None = None,
    on_result: Callable[[VetResult], None] | None = None,
    max_zero_progress: int = MAX_ZERO_PROGRESS_CANDIDATES,
) -> list[VetResult]:
    if not candidates:
        return []

    def run_worker(slice_: Sequence[VetCandidate], stream: Callable[[VetResult], None] | None) -> _WorkerOutcome:
        return _run_subprocess_worker(
            slice_,
            speed=speed,
            seed=seed,
            robot_uids=robot_uids,
            startup_timeout=startup_timeout,
            object_timeout=object_timeout,
            gso_manifest_path=gso_manifest_path,
            cache_dir=cache_dir,
            on_result=stream,
        )

    return _resume_vet(candidates, run_worker, on_result, max_zero_progress=max_zero_progress)


def _run_subprocess_worker(
    candidates: Sequence[VetCandidate],
    *,
    speed: float,
    seed: int,
    robot_uids: str,
    startup_timeout: float,
    object_timeout: float,
    gso_manifest_path: str | Path | None = None,
    cache_dir: Path | None = None,
    on_result: Callable[[VetResult], None] | None = None,
) -> _WorkerOutcome:
    cache = cache_dir or default_cache_dir()
    job = {
        "speed": speed,
        "seed": seed,
        "robot_uids": robot_uids,
        "manifest_path": str(gso_manifest_path or (cache / "gso_manifest.json")),
        "cache_dir": str(cache),
        "items": [[c.source, c.id] for c in candidates],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".vetjob.json", delete=False, encoding="utf-8") as f:
        json.dump(job, f)
        job_path = f.name

    stderr_file = tempfile.NamedTemporaryFile("w+b", suffix=".vetworker.err", delete=False)  # noqa: SIM115
    stderr_path = Path(stderr_file.name)

    proc = subprocess.Popen(
        [sys.executable, "-m", "sweep_bench.vetting", "--worker", job_path],
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        bufsize=0,
        env=os.environ.copy(),
    )
    results: list[VetResult] = []
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    buf = b""
    got_first = False
    env_ready = False
    startup_deadline = time.monotonic() + startup_timeout

    def outcome(status: Literal["done", "crash", "timeout"], detail: str) -> _WorkerOutcome:
        return _WorkerOutcome(results, status, detail, [] if status == "done" else _stderr_tail(stderr_path))

    try:
        while True:
            budget = object_timeout if got_first else startup_timeout
            wait = budget if got_first else max(0.0, startup_deadline - time.monotonic())
            ready, _, _ = select.select([fd], [], [], wait)
            if not ready:
                proc.kill()
                proc.wait()
                phase = "vetting" if got_first else ("first object" if env_ready else "env build")
                return outcome("timeout", f"no result within {budget:g}s ({phase})")
            chunk = os.read(fd, 65536)
            if not chunk:
                rc = proc.wait()
                if rc == 0 and len(results) == len(candidates):
                    return outcome("done", "ok")
                return outcome("crash", f"worker exited rc={rc} after {len(results)}/{len(candidates)}")
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith(_RESULT_TAG):
                    result = VetResult.from_dict(json.loads(line[len(_RESULT_TAG) :]))
                    results.append(result)
                    got_first = True
                    if on_result is not None:
                        on_result(result)
                elif line.startswith(_READY_TAG):
                    env_ready = True
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        stderr_file.close()
        with contextlib.suppress(OSError):
            os.unlink(job_path)
        with contextlib.suppress(OSError):
            os.unlink(stderr_path)


def _stderr_tail(path: Path, *, max_lines: int = 50, max_bytes: int = 8192) -> list[str]:
    try:
        with path.open("rb") as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - max_bytes))
            blob = f.read()
    except OSError:
        return []
    if size > max_bytes:
        _, _, blob = blob.partition(b"\n")
    lines = (line.rstrip() for line in blob.decode("utf-8", "replace").splitlines())
    return [line for line in lines if line.strip()][-max_lines:]


def _stderr_summary(tail: Sequence[str], *, max_lines: int = 3, max_chars: int = 300) -> str:
    joined = " | ".join(tail[-max_lines:])
    return joined if len(joined) <= max_chars else f"...{joined[-max_chars:]}"


def _worker_main(job_path: str) -> int:
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    speed = float(job["speed"])
    seed = int(job["seed"])
    robot_uids = str(job["robot_uids"])
    cache_dir = Path(str(job["cache_dir"]))
    items = job["items"]

    proc_by_id = {c.id: c for c in procedural_candidates()}
    lookup: dict[str, dict[str, VetCandidate]] = {"procedural": proc_by_id}
    if any(str(src) in MESH_SOURCES for src, _ in items):
        manifest = load_manifest(Path(str(job["manifest_path"])))
        for candidate in imported_mesh_candidates(manifest, cache_dir):
            lookup.setdefault(candidate.source, {})[candidate.id] = candidate
    candidates = [lookup[str(src)][str(cid)] for src, cid in items]

    env = make_vet_env(candidates[0], robot_uids=robot_uids)
    print(_READY_TAG, flush=True)
    try:
        for candidate in candidates:
            result = vet_candidate(env, candidate, speed=speed, seed=seed)
            print(f"{_RESULT_TAG}{json.dumps(result.to_dict())}", flush=True)
    finally:
        env.close()
    return 0


def summarize(results: Sequence[VetResult]) -> dict[str, object]:
    passed = [r for r in results if r.verdict == "pass"]
    rejected = [r for r in results if r.verdict == "reject"]
    by_source: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_source.setdefault(r.source, {"total": 0, "passed": 0})
        bucket["total"] += 1
        if r.verdict == "pass":
            bucket["passed"] += 1
    by_reason: dict[str, int] = {}
    for r in rejected:
        key = r.reason.split(";")[0].split("(")[0].strip() or "unknown"
        by_reason[key] = by_reason.get(key, 0) + 1
    return {
        "total": len(results),
        "passed": len(passed),
        "rejected": len(rejected),
        "by_source": by_source,
        "by_reason": by_reason,
    }


def build_manifest_dict(
    results: Sequence[VetResult],
    *,
    speed: float,
    seed: int,
    sources: Sequence[Source],
    cache_dir: Path,
    robot_uids: str = _DEFAULT_VET_ROBOT,
    object_timeout: float | None = _WORKER_OBJECT_TIMEOUT,
) -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "vet_speed": speed,
        "seed": seed,
        "robot": robot_uids,
        "object_timeout": object_timeout,
        "harness": harness_identity(),
        "sources": list(sources),
        "gso_cache_dir": home_relative(cache_dir.resolve()),
        "entries": [r.to_dict() for r in results],
        "summary": summarize(results),
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    write_json_atomic(path, manifest)


_RUN_CONFIG_KEY = "vet_run_config"


def _threshold_digest() -> str:
    values = {
        name: value
        for name, value in sorted(globals().items())
        if name.startswith("_") and not name.startswith("__") and type(value) in (int, float)
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]


def harness_identity() -> str:
    return f"v{VETTING_ALGO_VERSION}:{_threshold_digest()}"


@dataclass(frozen=True)
class VetRunConfig:
    speed: float
    seed: int
    robot_uids: str
    object_timeout: float | None = None
    harness: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            _RUN_CONFIG_KEY: {
                "speed": self.speed,
                "seed": self.seed,
                "robot_uids": self.robot_uids,
                "object_timeout": self.object_timeout,
                "harness": self.harness,
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> VetRunConfig:
        raw = data.get(_RUN_CONFIG_KEY)
        if not isinstance(raw, dict):
            raise ValueError(f"malformed {_RUN_CONFIG_KEY} line: {data!r}")
        payload: dict[str, object] = {str(k): v for k, v in raw.items()}
        speed, seed, robot = payload.get("speed"), payload.get("seed"), payload.get("robot_uids")
        if not isinstance(speed, (int, float)) or not isinstance(seed, int) or not isinstance(robot, str):
            raise ValueError(f"malformed {_RUN_CONFIG_KEY} payload: {payload!r}")
        timeout = payload.get("object_timeout")
        if timeout is not None and not isinstance(timeout, (int, float)):
            raise ValueError(f"malformed {_RUN_CONFIG_KEY} object_timeout: {timeout!r}")
        harness = payload.get("harness")
        if harness is not None and not isinstance(harness, str):
            raise ValueError(f"malformed {_RUN_CONFIG_KEY} harness: {harness!r}")
        return cls(
            speed=float(speed),
            seed=seed,
            robot_uids=robot,
            object_timeout=None if timeout is None else float(timeout),
            harness=harness,
        )

    def mismatches(self, recorded: VetRunConfig) -> list[str]:
        return [
            f"{f.name}: file has {getattr(recorded, f.name)!r}, this run {getattr(self, f.name)!r}"
            for f in fields(self)
            if getattr(self, f.name) != getattr(recorded, f.name)
            and not (f.name == "harness" and getattr(recorded, f.name) is None)
        ]


def read_results_config(path: Path) -> VetRunConfig | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or _RUN_CONFIG_KEY not in line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and _RUN_CONFIG_KEY in data:
                    return VetRunConfig.from_dict(data)
            except ValueError as e:
                logger.warning("skipping malformed run-config line %d in %s: %s", line_no, path, e)
    return None


def _claim_results_file(path: Path, config: VetRunConfig) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".claim")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(config.to_dict()) + "\n")
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            return False
        return True
    finally:
        tmp_path.unlink(missing_ok=True)


def discard_unused_results_claim(path: Path) -> bool:
    if not path.is_file():
        return False
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        return False
    try:
        VetRunConfig.from_dict(json.loads(lines[0]))
    except (ValueError, TypeError):
        return False
    path.unlink(missing_ok=True)
    return True


def check_results_config(path: Path, config: VetRunConfig) -> bool:
    if _claim_results_file(path, config):
        return True
    recorded = read_results_config(path)
    if recorded is None:
        if path.stat().st_size > 0:
            logger.warning(
                "%s records no run config (written before they were tracked); its verdicts cannot be checked "
                "against speed=%.2f seed=%d robot=%s",
                path,
                config.speed,
                config.seed,
                config.robot_uids,
            )
        append_jsonl_line(path, config.to_dict())
        recorded = read_results_config(path)
        if recorded is None:
            return False
    if recorded.harness is None and config.harness is not None and path.stat().st_size > 0:
        logger.warning(
            "%s records no vetting-harness identity (written before it was tracked); its verdicts "
            "cannot be checked against %s, so they may have been decided by different thresholds",
            path,
            config.harness,
        )
    mismatched = config.mismatches(recorded)
    if mismatched:
        raise ValueError(
            f"{path} records verdicts vetted under a different run config ({'; '.join(mismatched)}). "
            "Re-run with the recorded config, or point --results at a fresh file."
        )
    return False


def _ends_with_newline(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return True
    with path.open("rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) == b"\n"


def append_jsonl_line(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "" if _ends_with_newline(path) else "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + json.dumps(payload) + "\n")


def append_result_jsonl(path: Path, result: VetResult) -> None:
    append_jsonl_line(path, result.to_dict())


def read_results_jsonl(path: Path) -> dict[str, VetResult]:
    results: dict[str, VetResult] = {}
    if not path.is_file():
        return results
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError(f"expected a JSON object, got {type(data).__name__}")
                if _RUN_CONFIG_KEY in data:
                    continue
                result = VetResult.from_dict(data)
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                logger.warning("skipping malformed line %d in %s: %s", line_no, path, e)
                continue
            results[result.id] = result
    return results


def default_vetted_manifest_path(cache_dir: Path | None = None) -> Path:
    base = cache_dir or default_cache_dir()
    return base / "vetted_manifest.json"


def passing_ids(manifest: dict[str, object]) -> list[str]:
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        return []
    out: list[str] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        verdict: object = None
        eid: object = None
        for k, v in e.items():
            if k == "verdict":
                verdict = v
            elif k == "id":
                eid = v
        if verdict == "pass" and eid is not None:
            out.append(str(eid))
    return out


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(_worker_main(sys.argv[sys.argv.index("--worker") + 1]))
    raise SystemExit("sweep_bench.vetting is a library; run scripts/vet_objects.py")
