from __future__ import annotations

import functools
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import sapien
import sapien.physx
from pydantic import TypeAdapter

from sweep_bench.assets.catalog import ObjectSpec, register_object

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import trimesh
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

    from sweep_bench.assets.catalog import ObjectRole, ObjectSource

logger = logging.getLogger(__name__)


GSO_SOURCE = "gso"
GSO_LICENSE = "CC-BY 4.0"
GSO_ATTRIBUTION_COLLECTION = "Google Scanned Objects"

YCB_SOURCE = "ycb"
YCB_LICENSE = "CC-BY 4.0"
YCB_ATTRIBUTION_COLLECTION = "Yale-CMU-Berkeley (YCB) Object and Model Set"
YCB_SOURCE_URL = "https://registry.opendata.aws/ycb-benchmarks"

ROBOCASA_SOURCE = "robocasa"
ROBOCASA_LICENSE = "CC-BY 4.0"
ROBOCASA_ATTRIBUTION_COLLECTION = "RoboCasa"
ROBOCASA_SOURCE_URL = "https://github.com/robocasa/robocasa"

POLYHAVEN_SOURCE = "polyhaven"
POLYHAVEN_LICENSE = "CC0 1.0"
POLYHAVEN_ATTRIBUTION_COLLECTION = "Poly Haven"
POLYHAVEN_SOURCE_URL = "https://polyhaven.com"
POLYHAVEN_API = "https://api.polyhaven.com"
CC_BY_4_LICENSE_NAME = "Creative Commons Attribution 4.0 International"

_BELT_HALF_WIDTH = 0.08
_MAX_RIDER_FOOTPRINT = 0.07

_GRIPPER_OPENING = 0.08
_MAX_GRASP_WIDTH = 0.07

_MAX_SIZE_PRIOR_UPSCALE = 1.5
_SIZE_PRIOR_PATTERN = re.compile(r"(^|_)(dice|die|marble|marbles|domino|dominoes|lego|duplo|block|blocks|ball)(_|$)")


def has_size_prior(model_id: str) -> bool:
    return _SIZE_PRIOR_PATTERN.search(model_id) is not None


_MAX_TIP_ASPECT = 3.0

_MIN_RIDER_THICKNESS = 0.008

_DEFAULT_DENSITY = 1000.0

_MIN_HORIZONTAL_EXTENT = 1e-4
_MIN_COLLISION_VOLUME = 1e-9

_BODY_GAP_FRAC = 0.01
_MIN_BODY_VOLUME_FRAC = 0.02
_MAX_BODIES = 2
_PAIR_VOLUME_FRAC = 0.5

CollisionMethod = Literal["coacd", "convex_hull"]


def default_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "sweep_bench" / "gso"


def default_manifest_path(cache_dir: Path | None = None) -> Path:
    return (cache_dir or default_cache_dir()) / "gso_manifest.json"


def ycb_models_dir() -> Path:
    return maniskill_asset_dir() / "assets" / "mani_skill2_ycb" / "models"


def maniskill_asset_dir() -> Path:
    try:
        from mani_skill import ASSET_DIR
    except Exception:  # noqa: BLE001 — the fallback is the documented default path
        return Path.home() / ".maniskill" / "data"
    return Path(ASSET_DIR)


class MeshRejected(Exception):
    pass


class DecompositionUnavailable(Exception):
    pass


@dataclass(frozen=True)
class MeshMeasurement:
    scale: float
    native_center: tuple[float, float, float]
    native_half_extents: tuple[float, float, float]
    bbox: tuple[float, float, float]
    patch_radius: float
    grasp_width: float


def compute_scale(native_full_extents: tuple[float, float, float], *, model_id: str = "") -> float:
    fx, fy, _fz = native_full_extents
    min_h = min(fx, fy)
    max_h = max(fx, fy)
    if min_h < _MIN_HORIZONTAL_EXTENT or max_h < _MIN_HORIZONTAL_EXTENT:
        raise MeshRejected(f"degenerate horizontal extent (min={min_h:.2e} m, max={max_h:.2e} m)")
    scale = min(_GRIPPER_OPENING / max_h, _MAX_GRASP_WIDTH / min_h)
    if model_id and has_size_prior(model_id):
        return min(scale, _MAX_SIZE_PRIOR_UPSCALE)
    return scale


def measure_mesh(
    bounds_min: tuple[float, float, float], bounds_max: tuple[float, float, float], *, model_id: str = ""
) -> MeshMeasurement:
    center = (
        (bounds_min[0] + bounds_max[0]) / 2.0,
        (bounds_min[1] + bounds_max[1]) / 2.0,
        (bounds_min[2] + bounds_max[2]) / 2.0,
    )
    half = (
        (bounds_max[0] - bounds_min[0]) / 2.0,
        (bounds_max[1] - bounds_min[1]) / 2.0,
        (bounds_max[2] - bounds_min[2]) / 2.0,
    )

    scale = compute_scale((2.0 * half[0], 2.0 * half[1], 2.0 * half[2]), model_id=model_id)
    hx, hy, hz = scale * half[0], scale * half[1], scale * half[2]
    footprint = max(hx, hy)
    small = min(hx, hy)

    if footprint > _MAX_RIDER_FOOTPRINT + 1e-9:
        raise MeshRejected(f"footprint half-extent {footprint:.4f} m exceeds belt ceiling {_MAX_RIDER_FOOTPRINT} m")
    grasp_width = 2.0 * small
    if grasp_width > _MAX_GRASP_WIDTH + 1e-9:
        raise MeshRejected(f"grasp_width {grasp_width:.4f} m exceeds target {_MAX_GRASP_WIDTH} m")
    if hz > _MAX_TIP_ASPECT * small:
        raise MeshRejected(
            f"too tall for the belt: half-height {hz:.4f} m > {_MAX_TIP_ASPECT} x footprint {small:.4f} m"
        )
    return MeshMeasurement(
        scale=scale,
        native_center=center,
        native_half_extents=half,
        bbox=(hx, hy, hz),
        patch_radius=small,
        grasp_width=grasp_width,
    )


def load_mesh(path: Path) -> trimesh.Trimesh:
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.shape[0] == 0:
        raise MeshRejected(f"no triangle mesh loaded from {path}")
    return mesh


@dataclass(frozen=True)
class DecompositionResult:
    method: CollisionMethod
    parts: int
    native_volume: float


def _connected_convex_volume(mesh: trimesh.Trimesh) -> tuple[int, float]:
    import trimesh  # noqa: F401 — for typing parity; components are trimesh objects

    components = mesh.split(only_watertight=False)
    if len(components) == 0:
        components = [mesh]
    volume = 0.0
    parts = 0
    for comp in components:
        try:
            hull = comp.convex_hull
            vol = float(abs(hull.volume))
        except Exception:  # noqa: BLE001 — a degenerate component has no hull; skip it
            continue
        if vol <= 0.0:
            continue
        volume += vol
        parts += 1
    return parts, volume


def _points_hull_volume(points: object) -> float:
    import trimesh

    try:
        return float(abs(trimesh.PointCloud(points).convex_hull.volume))
    except Exception:  # noqa: BLE001 — a coplanar/degenerate cluster has no hull
        return 0.0


def _cluster_bodies(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[list[int]], list[np.ndarray]]:
    import numpy as np
    from trimesh.graph import connected_components

    merged = mesh.copy()
    merged.merge_vertices()
    vertices = np.asarray(merged.vertices, dtype=float)
    if len(vertices) == 0:
        return merged, [], []
    groups = connected_components(merged.edges_sorted, nodes=np.arange(len(vertices)), min_len=1)
    component_vertices = [np.asarray(g, dtype=int) for g in groups] or [np.arange(len(vertices))]
    lo = np.array([vertices[g].min(axis=0) for g in component_vertices])
    hi = np.array([vertices[g].max(axis=0) for g in component_vertices])
    eps = _BODY_GAP_FRAC * float((vertices.max(axis=0) - vertices.min(axis=0)).max())

    n = len(component_vertices)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    index = np.arange(n)
    for i in range(n):
        overlaps = np.all((lo <= hi[i] + eps) & (hi >= lo[i] - eps), axis=1) & (index > i)
        root_i = find(i)
        for j in np.nonzero(overlaps)[0]:
            root_j = find(int(j))
            if root_j != root_i:
                parent[root_j] = root_i
    bodies: dict[int, list[int]] = {}
    for i in range(n):
        bodies.setdefault(find(i), []).append(i)
    return merged, list(bodies.values()), component_vertices


def body_face_groups(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[np.ndarray]]:
    import numpy as np

    merged, bodies, component_vertices = _cluster_bodies(mesh)
    if not bodies:
        return merged, []
    vertices = np.asarray(merged.vertices, dtype=float)
    vertex_body = np.full(len(vertices), -1, dtype=int)
    for body_index, component_indices in enumerate(bodies):
        for c in component_indices:
            vertex_body[component_vertices[c]] = body_index
    faces = np.asarray(merged.faces, dtype=int)
    face_body = vertex_body[faces[:, 0]]

    selections: list[tuple[float, np.ndarray]] = []
    for body_index in range(len(bodies)):
        selected = np.nonzero(face_body == body_index)[0]
        if len(selected):
            volume = _points_hull_volume(vertices[np.unique(faces[selected])])
            selections.append((volume, selected))
    selections.sort(key=lambda kv: -kv[0])
    return merged, [sel for volume, sel in selections if volume > 0.0]


def body_volumes(mesh: trimesh.Trimesh) -> list[float]:
    import numpy as np

    merged, bodies, component_vertices = _cluster_bodies(mesh)
    if not bodies:
        return []
    vertices = np.asarray(merged.vertices, dtype=float)
    volumes = [_points_hull_volume(vertices[np.concatenate([component_vertices[i] for i in idxs])]) for idxs in bodies]
    return sorted((v for v in volumes if v > 0.0), reverse=True)


def check_single_body(mesh: trimesh.Trimesh) -> None:
    volumes = body_volumes(mesh)
    if not volumes:
        raise MeshRejected("no non-degenerate mesh body found")
    largest = volumes[0]
    bodies = [v for v in volumes if v >= _MIN_BODY_VOLUME_FRAC * largest]
    second_frac = bodies[1] / largest if len(bodies) > 1 else 0.0
    if len(bodies) > _MAX_BODIES or second_frac >= _PAIR_VOLUME_FRAC:
        raise MeshRejected(
            f"multi-component mesh: {len(bodies)} separate bodies, second/largest convex volume "
            f"{second_frac:.2f} (limits: <= {_MAX_BODIES} bodies, second < {_PAIR_VOLUME_FRAC}); "
            "loose objects in one scan merge into an ungraspable blob"
        )


def decompose_collision(
    mesh: trimesh.Trimesh,
    out_path: Path,
    *,
    method: CollisionMethod,
    coacd_params: dict[str, object] | None = None,
) -> DecompositionResult:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if method == "coacd":
        _coacd_to_ply(mesh, out_path, coacd_params or {})
    elif method == "convex_hull":
        _hull_to_ply(mesh, out_path)
    else:  # pragma: no cover — Literal keeps this unreachable
        raise ValueError(f"unknown collision method {method!r}")

    written = load_mesh(out_path)
    parts, native_volume = _connected_convex_volume(written)
    if parts == 0 or native_volume < _MIN_COLLISION_VOLUME:
        raise MeshRejected(f"degenerate collision ({parts} parts, volume {native_volume:.2e} m^3)")
    return DecompositionResult(method=method, parts=parts, native_volume=native_volume)


def measure_collision_file(path: Path) -> DecompositionResult:
    parts, native_volume = _connected_convex_volume(load_mesh(path))
    if parts == 0 or native_volume < _MIN_COLLISION_VOLUME:
        raise MeshRejected(f"degenerate collision ({parts} parts, volume {native_volume:.2e} m^3)")
    method: CollisionMethod = "coacd" if parts > 1 else "convex_hull"
    return DecompositionResult(method=method, parts=parts, native_volume=native_volume)


def _hull_to_ply(mesh: trimesh.Trimesh, out_path: Path) -> None:
    hull = mesh.convex_hull
    if float(abs(hull.volume)) < _MIN_COLLISION_VOLUME:
        raise MeshRejected(f"convex hull volume {float(abs(hull.volume)):.2e} m^3 below floor")
    out_path.write_bytes(hull.export(file_type="ply"))


def _coacd_to_ply(mesh: trimesh.Trimesh, out_path: Path, coacd_params: dict[str, object]) -> None:
    try:
        from sapien.wrapper.coacd import do_coacd
    except Exception as e:  # noqa: BLE001 — import wiring differs across sapien builds
        raise DecompositionUnavailable(f"cannot import sapien CoACD wrapper: {e}") from e

    staged = out_path.with_suffix(".coacd_input.obj")
    obj_data = mesh.export(file_type="obj")
    if isinstance(obj_data, str):
        staged.write_bytes(obj_data.encode())
    elif isinstance(obj_data, (bytes, bytearray)):
        staged.write_bytes(bytes(obj_data))
    else:
        raise MeshRejected(f"unexpected OBJ export type {type(obj_data).__name__}")
    try:
        result = do_coacd(str(staged), **coacd_params)
    except ModuleNotFoundError as e:
        raise DecompositionUnavailable(f"coacd package not installed: {e}") from e
    except Exception as e:  # noqa: BLE001 — CoACD raises plain Exception on failure
        raise MeshRejected(f"CoACD failed: {e}") from e
    finally:
        staged.unlink(missing_ok=True)
    Path(out_path).write_bytes(Path(result).read_bytes())
    Path(result).unlink(missing_ok=True)


def coacd_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("coacd") is not None
    except ModuleNotFoundError:
        return False


def mass_from_density(native_volume: float, scale: float, density: float = _DEFAULT_DENSITY) -> float:
    return density * native_volume * scale**3


def slug_for(name: str, prefix: str = GSO_SOURCE) -> str:
    body = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"{prefix}_{body}" if body else f"{prefix}_unnamed"


def object_role(role: str) -> ObjectRole:
    if role == "graspable_rider":
        return "graspable_rider"
    if role == "container":
        return "container"
    if role == "insert_socket":
        return "insert_socket"
    if role == "tool":
        return "tool"
    raise ValueError(f"manifest entry has unknown role {role!r}")


def object_source(source: str) -> ObjectSource:
    if source == GSO_SOURCE:
        return "gso"
    if source == YCB_SOURCE:
        return "ycb"
    if source == ROBOCASA_SOURCE:
        return "robocasa"
    if source == POLYHAVEN_SOURCE:
        return "polyhaven"
    raise ValueError(f"manifest entry has unknown source {source!r} (expected {GSO_SOURCE!r} or {YCB_SOURCE!r})")


@dataclass(frozen=True)
class ImportedMeshEntry:
    id: str
    name: str
    role: str
    source: str
    license: str
    attribution: str
    source_url: str
    visual_file: str
    collision_file: str
    collision_method: CollisionMethod
    collision_parts: int
    scale: float
    native_center: tuple[float, float, float]
    native_half_extents: tuple[float, float, float]
    bbox: tuple[float, float, float]
    mass: float
    density: float
    friction: float | None = None
    patch_radius: float | None = None
    grasp_width: float | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "source": self.source,
            "license": self.license,
            "attribution": self.attribution,
            "source_url": self.source_url,
            "visual_file": self.visual_file,
            "collision_file": self.collision_file,
            "collision_method": self.collision_method,
            "collision_parts": self.collision_parts,
            "scale": self.scale,
            "native_center": list(self.native_center),
            "native_half_extents": list(self.native_half_extents),
            "bbox": list(self.bbox),
            "mass": self.mass,
            "density": self.density,
            "friction": self.friction,
            "patch_radius": self.patch_radius,
            "grasp_width": self.grasp_width,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ImportedMeshEntry:
        return _ENTRY_SCHEMA.validate_python(data)

    def located(self, cache_root: Path) -> ImportedMeshEntry:
        return replace(
            self,
            visual_file=str(locate_mesh_file(self.visual_file, cache_root)),
            collision_file=str(locate_mesh_file(self.collision_file, cache_root)),
        )

    def has_mesh_files(self) -> bool:
        return Path(self.visual_file).is_file() and Path(self.collision_file).is_file()

    def to_spec(self) -> ObjectSpec:
        role = object_role(self.role)
        rides = role != "tool"
        pinched = role in ("graspable_rider", "tool")
        return ObjectSpec(
            id=self.id,
            role=object_role(self.role),
            source=object_source(self.source),
            builder_fn=functools.partial(
                build_imported_actor,
                visual_file=self.visual_file,
                collision_file=self.collision_file,
                scale=self.scale,
                center=self.native_center,
                density=self.density,
                friction=self.friction,
            ),
            bbox=self.bbox,
            mass=self.mass,
            patch_radius=self.patch_radius if rides else None,
            grasp_width=self.grasp_width if pinched else None,
            tags=self.tags,
        )


_ENTRY_SCHEMA = TypeAdapter(ImportedMeshEntry)

_CACHE_ROOT_NAME = "sweep_bench"
_MANISKILL_SUBTREES = (("assets", "mani_skill2_ycb"), ("scene_datasets", "robocasa_dataset"))


def locate_mesh_file(recorded: str, cache_root: Path) -> Path:
    path = Path(recorded)
    if path.is_file():
        return path
    parts = path.parts
    for subtree in _MANISKILL_SUBTREES:
        for start in range(len(parts) - len(subtree) + 1):
            if parts[start : start + len(subtree)] == subtree:
                return maniskill_asset_dir().joinpath(*parts[start:])
    if _CACHE_ROOT_NAME in parts:
        below = len(parts) - parts[::-1].index(_CACHE_ROOT_NAME)
        return cache_root.joinpath(*parts[below:])
    return path


def ensure_visual_textures(visual_file: str | Path) -> list[Path]:
    obj_path = Path(visual_file)
    if not obj_path.is_file() or obj_path.suffix.lower() != ".obj":
        return []
    model_root = obj_path.parent.parent.resolve()
    textures_dir = model_root / "materials" / "textures"
    created: list[Path] = []
    for mtl_name in re.findall(r"^mtllib\s+(.+)$", obj_path.read_text(errors="ignore"), re.MULTILINE):
        mtl_path = obj_path.parent / mtl_name.strip()
        if not within_root(model_root, mtl_path):
            raise MeshRejected(
                f"visual {obj_path} names material {mtl_name.strip()!r}, which resolves outside the model "
                f"directory {model_root}; refusing to read or repair it"
            )
        if not mtl_path.is_file():
            continue
        created.extend(_repair_material(mtl_path, obj_path.parent, textures_dir, model_root))
    return created


_DIFFUSE_MAP_DIRECTIVE = "map_kd"
_STRIPPED_MARKER = "# dropped by sweep_bench (not shipped in the archive): "


def within_root(root: Path, path: Path) -> bool:
    resolved = path.resolve()
    return resolved == root or root in resolved.parents


def _repair_material(mtl_path: Path, obj_dir: Path, textures_dir: Path, model_root: Path) -> list[Path]:
    created: list[Path] = []
    lines = mtl_path.read_text(errors="ignore").splitlines(keepends=True)
    stripped = False
    for i, line in enumerate(lines):
        match = re.match(r"^(map_\w+)\s+(.+?)\s*$", line)
        if match is None:
            continue
        directive, tex_ref = match.group(1), match.group(2)
        target = obj_dir / tex_ref
        if not within_root(model_root, target):
            raise MeshRejected(
                f"visual material {mtl_path} references texture {tex_ref!r}, which resolves outside the "
                f"model directory {model_root}; refusing to create or use it"
            )
        if target.exists():
            continue
        source = textures_dir / Path(tex_ref).name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.path.relpath(source, target.parent))
            created.append(target)
            continue
        if directive.lower() == _DIFFUSE_MAP_DIRECTIVE:
            raise MeshRejected(
                f"visual material {mtl_path} references diffuse texture {tex_ref!r} that exists neither "
                f"next to the OBJ nor in {textures_dir}"
            )
        logger.warning(
            "%s: stripping dead %s reference %r (absent from the archive; shading fidelity only)",
            mtl_path.parent.parent.name,
            directive,
            tex_ref,
        )
        lines[i] = _STRIPPED_MARKER + line.lstrip()
        stripped = True
    if stripped:
        mtl_path.write_text("".join(lines), encoding="utf-8")
    return created


def build_imported_actor(
    scene: ManiSkillScene,
    *,
    visual_file: str,
    collision_file: str,
    scale: float,
    center: tuple[float, float, float],
    density: float,
    friction: float | None,
) -> ActorBuilder:
    for label, path in (("visual", visual_file), ("collision", collision_file)):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"GSO {label} mesh missing: {path}. Re-run scripts/import_objects.py to rebuild the asset cache."
            )
    ensure_visual_textures(visual_file)
    material = None
    if friction is not None:
        material = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    offset = sapien.Pose(p=[-center[0] * scale, -center[1] * scale, -center[2] * scale])
    s = (scale, scale, scale)

    builder = scene.create_actor_builder()
    builder.add_multiple_convex_collisions_from_file(
        filename=collision_file,
        scale=s,
        pose=offset,
        density=density,
        material=material,
        decomposition="none",
    )
    builder.add_visual_from_file(filename=visual_file, scale=s, pose=offset)
    return builder


MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RejectedModel:
    name: str
    id: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "id": self.id, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RejectedModel:
        return _REJECTION_SCHEMA.validate_python(data)


_REJECTION_SCHEMA = TypeAdapter(RejectedModel)


@dataclass(frozen=True)
class GsoManifest:
    schema_version: int
    source: str = GSO_SOURCE
    license: str = GSO_LICENSE
    cache_dir: str = ""
    entries: tuple[ImportedMeshEntry, ...] = ()
    rejections: tuple[RejectedModel, ...] = ()
    pending_revet: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "license": self.license,
            "cache_dir": self.cache_dir,
            "entries": [e.to_dict() for e in self.entries],
            "rejections": [r.to_dict() for r in self.rejections],
            "pending_revet": list(self.pending_revet),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GsoManifest:
        version = data.get("schema_version")
        if version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema_version {version!r} (expected {MANIFEST_SCHEMA_VERSION})")
        return _MANIFEST_SCHEMA.validate_python(data)


_MANIFEST_SCHEMA = TypeAdapter(GsoManifest)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_manifest(manifest_path: str | Path) -> GsoManifest:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"GSO manifest not found: {path}. Generate it with "
            f"`uv run --no-sync python scripts/import_objects.py --out {path}`."
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"manifest {path} is not a JSON object")
    return GsoManifest.from_dict(data)


def load_located_entries(manifest_path: str | Path | None = None) -> tuple[ImportedMeshEntry, ...]:
    path = Path(manifest_path or default_manifest_path())
    cache_root = path.absolute().parent.parent
    return tuple(entry.located(cache_root) for entry in load_manifest(path).entries)


def load_gso_specs(manifest_path: str | Path | None = None) -> list[ObjectSpec]:
    return [entry.to_spec() for entry in load_located_entries(manifest_path)]


def register_gso_objects(
    manifest_path: str | Path | None = None,
    register: Callable[[ObjectSpec], None] = register_object,
) -> int:
    specs = load_gso_specs(manifest_path)
    for spec in specs:
        register(spec)
    return len(specs)
