from __future__ import annotations

import json
import math
import pathlib
from typing import TypedDict

CENTER = 0.25
SCALE = 0.15
REQUEST_FIELD = "belt_speed"
METADATA_KEY = "speed_conditioned"
CHECKPOINT_STAMP = "bench_speed_conditioning.json"
STATE_DIM = 9


class Normalisation(TypedDict):
    center: float
    scale: float


def normalize(speed: float, norm: Normalisation) -> float:
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed < 0:
        raise ValueError(f"belt speed must be a finite |velocity| >= 0 in m/s, got {speed!r}")
    return (speed - norm["center"]) / norm["scale"]


def read_stamp(run_dir: pathlib.Path) -> Normalisation | None:
    path = pathlib.Path(run_dir) / CHECKPOINT_STAMP
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not all(isinstance(data.get(k), float) for k in ("center", "scale")):
        raise ValueError(f"{path} is not a valid {CHECKPOINT_STAMP}: {data!r}")
    return {"center": data["center"], "scale": data["scale"]}


def stamp_run_dir(run_dir: pathlib.Path, conditioned: bool, *, has_checkpoints: bool) -> None:
    recorded = read_stamp(run_dir)
    if not conditioned:
        if recorded is not None:
            raise RuntimeError(f"{run_dir} was trained speed-conditioned; this run is not. Use a fresh run dir.")
        return
    norm: Normalisation = {"center": CENTER, "scale": SCALE}
    if recorded is not None:
        if recorded != norm:
            raise RuntimeError(f"{run_dir} was trained with speed normalisation {recorded}, this run uses {norm}")
        return
    if has_checkpoints:
        raise RuntimeError(f"{run_dir} holds checkpoints trained without the belt speed; use a fresh run dir")
    path = pathlib.Path(run_dir) / CHECKPOINT_STAMP
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(norm))
    tmp.replace(path)


def check_serve(run_dir: pathlib.Path, configured: bool, state_dim: int) -> Normalisation | None:
    norm = read_stamp(run_dir)
    trained = norm is not None
    if state_dim != (STATE_DIM if trained else STATE_DIM - 1):
        raise RuntimeError(
            f"{run_dir}: config state_dim {state_dim} contradicts its {CHECKPOINT_STAMP} (speed-conditioned: {trained})"
        )
    if trained != configured:
        raise RuntimeError(
            f"checkpoint in {run_dir} was trained {'with' if trained else 'without'} the belt speed in its state, "
            f"but the server is started {'with' if configured else 'without'} --speed-conditioned"
        )
    return norm
