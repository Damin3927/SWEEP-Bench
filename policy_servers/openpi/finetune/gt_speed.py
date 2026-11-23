from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib

ENV_VAR = "BENCH_GT_SPEED"
LIST_ENV_VAR = "BENCH_GT_SPEED_LIST"
CHECKPOINT_STAMP = "bench_gt_speed.json"
METADATA_KEY = "speed_conditioned"
REQUEST_KEY = "belt_speed"

STATE_SLOT = 8
CENTER = 0.25
SCALE = 0.15


def enabled_from_env() -> bool:
    raw = os.environ.get(ENV_VAR, "").strip()
    if raw not in ("", "0", "1"):
        raise ValueError(f"{ENV_VAR} must be empty, 0 or 1, got {raw!r}")
    return raw == "1"


def encode(speed: float) -> float:
    return (speed - CENTER) / SCALE


def check_speed(speed: object, source: str) -> float:
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed):
        raise ValueError(f"{source}: belt speed must be a finite number in m/s, got {speed!r}")
    if not 0.0 <= speed < 1.0:
        raise ValueError(f"{source}: belt speed {speed!r} m/s is outside [0, 1); expected unsigned m/s")
    return float(speed)


def load_episode_speeds(path: str | pathlib.Path, num_episodes: int) -> tuple[float, ...]:
    data = json.loads(pathlib.Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must hold a JSON list of speeds indexed by episode_index, got {type(data).__name__}")
    if len(data) != num_episodes:
        raise ValueError(f"{path} lists {len(data)} speeds but the dataset has {num_episodes} episodes")
    return tuple(check_speed(v, f"{path}[{i}]") for i, v in enumerate(data))


def list_sha256(path: str | pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _stamp_record(list_path: str | pathlib.Path) -> dict:
    return {
        METADATA_KEY: True,
        "state_slot": STATE_SLOT,
        "center": CENTER,
        "scale": SCALE,
        "speed_list": str(list_path),
        "speed_list_sha256": list_sha256(list_path),
    }


def stamp_run_dir(
    run_dir: pathlib.Path, enabled: bool, list_path: str | None, *, existing_steps: tuple[int, ...]
) -> None:
    path = pathlib.Path(run_dir) / CHECKPOINT_STAMP
    recorded = read_stamp(path) if path.is_file() else None
    if not enabled:
        if recorded is not None:
            raise RuntimeError(f"{run_dir} holds GT-speed checkpoints, but {ENV_VAR} is off. Use a fresh --exp-name.")
        return
    if not list_path:
        raise RuntimeError(f"{ENV_VAR}=1 needs {LIST_ENV_VAR}, the per-episode speed list")
    record = _stamp_record(list_path)
    if recorded is not None:
        if {k: recorded[k] for k in ("state_slot", "center", "scale", "speed_list_sha256")} != {
            k: record[k] for k in ("state_slot", "center", "scale", "speed_list_sha256")
        }:
            raise RuntimeError(
                f"{run_dir} was trained with GT speed {recorded}, but this run uses {record}. Use a fresh --exp-name."
            )
        return
    if existing_steps:
        raise RuntimeError(
            f"{run_dir} already holds checkpoints {existing_steps} without {CHECKPOINT_STAMP}, so they were trained "
            f"without GT speed; resuming them with it would mix the two. Use a fresh --exp-name."
        )
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(path)
    print(f"[bench] stamped {path} with GT speed {record}", flush=True)


def read_stamp(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get(METADATA_KEY) is not True:
        raise ValueError(f"{path} is not a valid {CHECKPOINT_STAMP}: {data!r}")
    if (data.get("state_slot"), data.get("center"), data.get("scale")) != (STATE_SLOT, CENTER, SCALE):
        raise ValueError(
            f"{path} encodes the speed as slot {data.get('state_slot')}, (v - {data.get('center')}) / "
            f"{data.get('scale')}; this code uses slot {STATE_SLOT}, (v - {CENTER}) / {SCALE}"
        )
    return data


def checkpoint_conditioned(ckpt_dir: str | pathlib.Path) -> bool:
    path = pathlib.Path(ckpt_dir)
    for candidate in (path, *path.parents):
        stamp = candidate / CHECKPOINT_STAMP
        if stamp.is_file():
            read_stamp(stamp)
            return True
    return False


def check_serve(ckpt_dir: str | pathlib.Path, configured: bool) -> None:
    trained = checkpoint_conditioned(ckpt_dir)
    if trained != configured:
        raise RuntimeError(
            f"checkpoint {ckpt_dir} was trained {'with' if trained else 'without'} GT speed (per "
            f"{CHECKPOINT_STAMP}{'' if trained else ' being absent'}), but {ENV_VAR} is "
            f"{'1' if configured else 'off'}. Serve it with {ENV_VAR}={'1' if trained else '0'}."
        )
