from __future__ import annotations

import json
import os
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

ENV_VAR = "BENCH_EPISODE_UNIFORM"
CHECKPOINT_STAMP = "bench_train_sampler.json"
SAMPLER = "episode_uniform"


def enabled_from_env() -> bool:
    raw = os.environ.get(ENV_VAR, "").strip()
    if raw not in ("", "0", "1"):
        raise ValueError(f"{ENV_VAR} must be empty, 0 or 1, got {raw!r}")
    return raw == "1"


def frame_weights(lengths: Sequence[int]) -> np.ndarray:
    lengths = np.asarray(lengths)
    if lengths.ndim != 1 or lengths.size == 0 or lengths.dtype.kind not in "iu" or np.any(lengths <= 0):
        raise ValueError(f"episode lengths must be a non-empty list of positive integers, got {lengths!r}")
    weights = np.repeat(1.0 / lengths, lengths)
    per_episode = np.add.reduceat(weights, np.cumsum(lengths) - lengths)
    if not np.allclose(per_episode, 1.0) or not np.isclose(weights.sum(), lengths.size):
        raise AssertionError(f"episode weights do not sum to 1 each: {per_episode}")
    return weights


def sampler(lengths: Sequence[int], num_frames: int, seed: int) -> torch.utils.data.WeightedRandomSampler:
    weights = frame_weights(lengths)
    if weights.size != num_frames:
        raise ValueError(f"the episode lengths cover {weights.size} frames, but the dataset has {num_frames}")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=num_frames, replacement=True, generator=generator
    )


def stamp_run_dir(run_dir: pathlib.Path, enabled: bool, *, existing_steps: tuple[int, ...]) -> None:
    path = pathlib.Path(run_dir) / CHECKPOINT_STAMP
    recorded = read_stamp(path) if path.is_file() else None
    if not enabled:
        if recorded is not None:
            raise RuntimeError(
                f"{run_dir} was trained with the {recorded} sampler, but {ENV_VAR} is off. Use a fresh --exp-name."
            )
        return
    if recorded is not None:
        return
    if existing_steps:
        raise RuntimeError(
            f"{run_dir} already holds checkpoints {existing_steps} without {CHECKPOINT_STAMP}, so they were trained "
            f"frame-uniform; resuming them episode-uniform would mix the two. Use a fresh --exp-name."
        )
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"sampler": SAMPLER}))
    tmp.replace(path)
    print(f"[bench] stamped {path} with the {SAMPLER} sampler", flush=True)


def read_stamp(path: pathlib.Path) -> str:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("sampler") != SAMPLER:
        raise ValueError(f"{path} is not a valid {CHECKPOINT_STAMP}: {data!r}")
    return data["sampler"]
