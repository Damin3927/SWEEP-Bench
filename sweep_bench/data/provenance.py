from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, cast

from sweep_bench.data.vintage import as_json_object, load_info

if TYPE_CHECKING:
    from pathlib import Path

type DrawScalar = bool | int | float | str
type DrawValue = DrawScalar | list[DrawScalar]

_SCALARS = (bool, int, float, str)

PROVENANCE_RELPATH = "meta/bench_episode_provenance.jsonl"

COLLECTION_INFO_KEY = "bench_collection"


@dataclass(frozen=True)
class CollectionStamp:
    task_id: str
    robot: str
    scene_style: str
    scene_idx: int
    workspace_offset: list[float] | None
    shader_pack: str
    control_mode: str
    num_envs: int
    randomize: bool
    object_pool: list[str]
    main_seed: list[int]
    max_steps: int
    filter_success: bool
    reconfigure_every: int
    speed_bins: int
    expert_speed_scale: float | None
    velrand_floor_ramp: list[float] | None
    platform_style: str = "box"
    background_pool: list[list[str | int]] | None = None
    accel_bins: int = 1


class _CollectionStampRequired(TypedDict):
    task_id: str
    robot: str
    scene_style: str
    scene_idx: int
    shader_pack: str
    control_mode: str
    num_envs: int
    randomize: bool
    object_pool: list[str]
    main_seed: list[int]
    max_steps: int
    filter_success: bool
    reconfigure_every: int
    speed_bins: int
    expert_speed_scale: float | None


class CollectionStampJson(_CollectionStampRequired, total=False):
    velrand_floor_ramp: list[float]
    workspace_offset: list[float]
    platform_style: str
    background_pool: list[list[str | int]]
    accel_bins: int


def collection_stamp_json(stamp: CollectionStamp) -> CollectionStampJson:
    blob = dataclasses.asdict(stamp)
    for optional in ("velrand_floor_ramp", "workspace_offset", "background_pool"):
        if blob.get(optional) is None:
            blob.pop(optional, None)
    if blob.get("platform_style") == "box":
        blob.pop("platform_style")
    if blob.get("accel_bins") == 1:
        blob.pop("accel_bins")
    return cast("CollectionStampJson", blob)


@dataclass(frozen=True)
class EpisodeProvenance:
    env_index: int
    reset_index: int
    reconfigure_index: int
    env_seed: int
    draw_seed: int
    prompt: str
    object_id: str | None
    belt_speed: float
    belt_direction: str
    belt_velocities: list[float]
    randomization: dict[str, DrawValue]
    velocity_randomization: dict[str, DrawValue]
    initial_poses: dict[str, list[float]]
    latched_draws: dict[str, DrawValue]
    scene_style: str | None = None
    scene_idx: int | None = None

    workspace_offset: list[float] | None = None

    belt_accel: float | None = None
    belt_center_speed: float | None = None


def config_snapshot(config: object) -> dict[str, DrawValue]:
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        raise TypeError(f"config_snapshot needs a dataclass INSTANCE, got {type(config).__name__}")
    snapshot: dict[str, DrawValue] = {}
    for field in dataclasses.fields(config):
        snapshot[field.name] = _as_draw_value(getattr(config, field.name), f"{type(config).__name__}.{field.name}")
    return snapshot


def _as_draw_value(value: object, where: str) -> DrawValue:
    if isinstance(value, _SCALARS):
        return value
    if isinstance(value, (tuple, list)):
        items = list(value)
        bad = [type(v).__name__ for v in items if not isinstance(v, _SCALARS)]
        if bad:
            raise TypeError(f"{where} holds a sequence of {sorted(set(bad))}, which is not JSON provenance")
        return items
    raise TypeError(
        f"{where} is a {type(value).__name__}, which this provenance record cannot represent. "
        f"Teach `provenance.config_snapshot` about it rather than letting the knob go unrecorded."
    )


def append_episode_provenance(dataset_root: Path, episode_index: int, provenance: EpisodeProvenance) -> None:
    path = dataset_root / PROVENANCE_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"episode_index": int(episode_index), **dataclasses.asdict(provenance)}
    for optional in (*_EPISODE_BACKGROUND_KEYS, *_EPISODE_ACCEL_KEYS):
        if row[optional] is None:
            del row[optional]
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def read_episode_provenance(dataset_root: Path) -> dict[int, EpisodeProvenance] | None:
    path = dataset_root / PROVENANCE_RELPATH
    if not path.is_file():
        return None
    episodes: dict[int, EpisodeProvenance] = {}
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}:{lineno}"
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where} is not JSON, so this dataset's episode provenance is damaged: {exc}") from exc
        row = as_json_object(parsed)
        if row is None:
            raise ValueError(f"{where} holds a JSON {type(parsed).__name__}, but every line here is an object.")
        index = _req(row, "episode_index", int, where)
        if index in episodes:
            raise ValueError(f"{where} repeats episode_index {index}; the sidecar must hold one line per episode.")
        episodes[index] = _episode_from_row(row, where)
    return episodes


def read_collection_stamp(dataset_root: Path) -> CollectionStamp | None:
    info = load_info(dataset_root)
    if info is None or COLLECTION_INFO_KEY not in info:
        return None
    where = f"{dataset_root}/meta/info.json[{COLLECTION_INFO_KEY!r}]"
    stamp = as_json_object(info[COLLECTION_INFO_KEY])
    if stamp is None:
        raise ValueError(f"{where} is a {type(info[COLLECTION_INFO_KEY]).__name__}, expected an object.")
    return CollectionStamp(
        task_id=_req(stamp, "task_id", str, where),
        robot=_req(stamp, "robot", str, where),
        scene_style=_req(stamp, "scene_style", str, where),
        scene_idx=_req(stamp, "scene_idx", int, where),
        workspace_offset=_offset_triple(stamp, where),
        shader_pack=_req(stamp, "shader_pack", str, where),
        control_mode=_req(stamp, "control_mode", str, where),
        num_envs=_req(stamp, "num_envs", int, where),
        randomize=_req(stamp, "randomize", bool, where),
        object_pool=_str_list(stamp, "object_pool", where),
        main_seed=_int_list(stamp, "main_seed", where),
        max_steps=_req(stamp, "max_steps", int, where),
        filter_success=_req(stamp, "filter_success", bool, where),
        reconfigure_every=_req(stamp, "reconfigure_every", int, where),
        speed_bins=_req(stamp, "speed_bins", int, where),
        expert_speed_scale=_opt_float(stamp, "expert_speed_scale", where),
        velrand_floor_ramp=_ramp_triple(stamp, where),
        platform_style=_platform_style(stamp, where),
        background_pool=_background_pool(stamp, where),
        accel_bins=_accel_bins(stamp, where),
    )


def _background_pool(stamp: dict[str, object], where: str) -> list[list[str | int]] | None:
    if "background_pool" not in stamp:
        return None
    pool = stamp["background_pool"]
    if not isinstance(pool, list) or not pool:
        raise _damaged(where, "background_pool", f"expected a non-empty list, got {pool!r}")
    pairs: list[list[str | int]] = []
    for pair in pool:
        if not (
            isinstance(pair, list)
            and len(pair) == 2
            and isinstance(pair[0], str)
            and isinstance(pair[1], int)
            and not isinstance(pair[1], bool)
        ):
            raise _damaged(where, "background_pool", f"holds {pair!r}, not a [scene_style, scene_idx] pair")
        pairs.append([pair[0], pair[1]])
    return pairs


def _accel_bins(stamp: dict[str, object], where: str) -> int:
    if "accel_bins" not in stamp:
        return 1
    value = _req(stamp, "accel_bins", int, where)
    if value <= 1:
        raise _damaged(where, "accel_bins", f"is {value}; the writer records only counts above 1")
    return value


def _platform_style(stamp: dict[str, object], where: str) -> str:
    if "platform_style" not in stamp:
        return "box"
    value = _req(stamp, "platform_style", str, where)
    if value != "table":
        raise _damaged(where, "platform_style", f"is {value!r}; the writer records only 'table'")
    return value


def _ramp_triple(stamp: dict[str, object], where: str) -> list[float] | None:
    if "velrand_floor_ramp" not in stamp:
        return None
    if stamp["velrand_floor_ramp"] is None:
        raise _damaged(
            where,
            "velrand_floor_ramp",
            "is present but null; the writer always emits [top, v_lo, v_hi], and an absent KEY "
            "is what means the dataset predates the ramp. A written null cannot be read as legacy",
        )
    values = _float_list(stamp, "velrand_floor_ramp", where)
    if len(values) != 3:
        raise _damaged(
            where,
            "velrand_floor_ramp",
            f"expected the declared [top, v_lo, v_hi], got {len(values)} value(s); a stamp that "
            "cannot name the ramp cannot identify the actions it produced",
        )
    top, v_lo, v_hi = values
    if not all(math.isfinite(v) for v in values):
        raise _damaged(where, "velrand_floor_ramp", f"contains a non-finite value: {values}")
    if not v_lo < v_hi:
        raise _damaged(
            where,
            "velrand_floor_ramp",
            f"declares v_lo={v_lo} and v_hi={v_hi}; the ramp interpolates between them, so equal "
            "endpoints divide by zero and reversed endpoints invert it",
        )
    del top
    return values


def _offset_triple(stamp: dict[str, object], where: str) -> list[float] | None:
    if "workspace_offset" not in stamp:
        return None
    if stamp["workspace_offset"] is None:
        raise _damaged(where, "workspace_offset", "is present but null; an absent key is what marks a legacy stamp")
    values = _float_list(stamp, "workspace_offset", where)
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise _damaged(where, "workspace_offset", f"expected three finite numbers, got {values}")
    return values


def _episode_from_row(row: dict[str, object], where: str) -> EpisodeProvenance:
    return EpisodeProvenance(
        env_index=_req(row, "env_index", int, where),
        reset_index=_req(row, "reset_index", int, where),
        reconfigure_index=_req(row, "reconfigure_index", int, where),
        env_seed=_req(row, "env_seed", int, where),
        draw_seed=_req(row, "draw_seed", int, where),
        prompt=_req(row, "prompt", str, where),
        object_id=_opt_str(row, "object_id", where),
        belt_speed=_float(row, "belt_speed", where),
        belt_direction=_req(row, "belt_direction", str, where),
        belt_velocities=_float_list(row, "belt_velocities", where),
        randomization=_draw_map(row, "randomization", where),
        velocity_randomization=_draw_map(row, "velocity_randomization", where),
        initial_poses=_pose_map(row, "initial_poses", where),
        latched_draws=_draw_map(row, "latched_draws", where),
        **_episode_background(row, where),
        **_episode_accel(row, where),
    )


_EPISODE_BACKGROUND_KEYS = ("scene_style", "scene_idx", "workspace_offset")
_EPISODE_ACCEL_KEYS = ("belt_accel", "belt_center_speed")


class _EpisodeAccel(TypedDict, total=False):
    belt_accel: float
    belt_center_speed: float


def _episode_accel(row: dict[str, object], where: str) -> _EpisodeAccel:
    present = [k for k in _EPISODE_ACCEL_KEYS if row.get(k) is not None]
    if not present:
        return {}
    if len(present) != len(_EPISODE_ACCEL_KEYS):
        raise _damaged(where, present[0], f"is present without {set(_EPISODE_ACCEL_KEYS) - set(present)}")
    accel, center = _float(row, "belt_accel", where), _float(row, "belt_center_speed", where)
    if not (math.isfinite(accel) and math.isfinite(center) and center > 0.0):
        raise _damaged(
            where, "belt_accel", f"expected a finite acceleration and a positive centre speed, got {accel}, {center}"
        )
    return {"belt_accel": accel, "belt_center_speed": center}


class _EpisodeBackground(TypedDict, total=False):
    scene_style: str
    scene_idx: int
    workspace_offset: list[float]


def _episode_background(row: dict[str, object], where: str) -> _EpisodeBackground:
    present = [k for k in _EPISODE_BACKGROUND_KEYS if row.get(k) is not None]
    if not present:
        return {}
    if len(present) != len(_EPISODE_BACKGROUND_KEYS):
        raise _damaged(where, "scene_style", f"the background keys come together; this row has only {present}")
    offset = _float_list(row, "workspace_offset", where)
    if len(offset) != 3:
        raise _damaged(where, "workspace_offset", f"expected three numbers, got {offset}")
    return {
        "scene_style": _req(row, "scene_style", str, where),
        "scene_idx": _req(row, "scene_idx", int, where),
        "workspace_offset": offset,
    }


def _damaged(where: str, key: str, detail: str) -> ValueError:
    return ValueError(
        f"{where} has a malformed {key!r}: {detail}. This is damaged provenance, not a pre-provenance dataset."
    )


def _req[T: (bool, int, str)](row: dict[str, object], key: str, kind: type[T], where: str) -> T:
    value = row.get(key)
    if not isinstance(value, kind) or (kind is not bool and isinstance(value, bool)):
        raise _damaged(where, key, f"expected {kind.__name__}, got {type(value).__name__}")
    return value


def _float(row: dict[str, object], key: str, where: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _damaged(where, key, f"expected a number, got {type(value).__name__}")
    return float(value)


def _opt_float(row: dict[str, object], key: str, where: str) -> float | None:
    if row.get(key) is None:
        if key not in row:
            raise _damaged(where, key, "missing")
        return None
    return _float(row, key, where)


def _opt_str(row: dict[str, object], key: str, where: str) -> str | None:
    if row.get(key) is None:
        if key not in row:
            raise _damaged(where, key, "missing")
        return None
    return _req(row, key, str, where)


def _sequence(row: dict[str, object], key: str, where: str) -> list[object]:
    value = row.get(key)
    if not isinstance(value, list):
        raise _damaged(where, key, f"expected a list, got {type(value).__name__}")
    return list(value)


def _float_list(row: dict[str, object], key: str, where: str) -> list[float]:
    items = _sequence(row, key, where)
    bad = [type(v).__name__ for v in items if isinstance(v, bool) or not isinstance(v, (int, float))]
    if bad:
        raise _damaged(where, key, f"expected numbers, got {sorted(set(bad))}")
    return [float(v) for v in items if isinstance(v, (int, float))]


def _int_list(row: dict[str, object], key: str, where: str) -> list[int]:
    items = _sequence(row, key, where)
    bad = [type(v).__name__ for v in items if isinstance(v, bool) or not isinstance(v, int)]
    if bad:
        raise _damaged(where, key, f"expected ints, got {sorted(set(bad))}")
    return [v for v in items if isinstance(v, int)]


def _str_list(row: dict[str, object], key: str, where: str) -> list[str]:
    items = _sequence(row, key, where)
    bad = [type(v).__name__ for v in items if not isinstance(v, str)]
    if bad:
        raise _damaged(where, key, f"expected strings, got {sorted(set(bad))}")
    return [v for v in items if isinstance(v, str)]


def _mapping(row: dict[str, object], key: str, where: str) -> dict[str, object]:
    mapping = as_json_object(row.get(key))
    if mapping is None:
        raise _damaged(where, key, f"expected an object, got {type(row.get(key)).__name__}")
    return mapping


def _draw_map(row: dict[str, object], key: str, where: str) -> dict[str, DrawValue]:
    out: dict[str, DrawValue] = {}
    for name, value in _mapping(row, key, where).items():
        try:
            out[name] = _as_draw_value(value, f"{key}.{name}")
        except TypeError as exc:
            raise _damaged(where, key, str(exc)) from exc
    return out


def _pose_map(row: dict[str, object], key: str, where: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for name, value in _mapping(row, key, where).items():
        if not isinstance(value, list) or len(value) != 7:
            raise _damaged(where, key, f"{name} is not a 7-element pose")
        bad = [type(v).__name__ for v in value if isinstance(v, bool) or not isinstance(v, (int, float))]
        if bad:
            raise _damaged(where, key, f"{name} holds {sorted(set(bad))} where a pose holds numbers")
        out[name] = [float(v) for v in value if isinstance(v, (int, float))]
    return out
