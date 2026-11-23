from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import pathlib
import sys
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvf

if TYPE_CHECKING:
    import types

HERE = pathlib.Path(__file__).resolve().parent
UPSTREAM = HERE.parent / "upstream"
REPO_ROOT = HERE.parents[2]

os.environ["PRISMATIC_ROBOT_PLATFORM"] = "BENCH"
if "prismatic" in sys.modules:
    raise RuntimeError("bench_common must be imported before any prismatic module")
sys.path.insert(0, str(UPSTREAM))

import prismatic.vla.constants as _constants  # noqa: E402

if getattr(_constants, "ROBOT_PLATFORM", None) != "BENCH":
    raise RuntimeError("upstream/ lacks the BENCH platform; run `make patch` in policy_servers/openvlaoft")


def _int_constant(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected an int constant, got {value!r}")
    return value


NUM_ACTIONS_CHUNK = _int_constant(_constants.NUM_ACTIONS_CHUNK)
ACTION_DIM = _int_constant(_constants.ACTION_DIM)
PROPRIO_DIM = _int_constant(_constants.PROPRIO_DIM)
ARM_DIM = 7

DATASET_SCHEMA = "droid_panda_v3"
UNNORM_KEY = "bench"
STATS_FILE = "dataset_statistics.json"
RUN_CONFIG_FILE = "bench_run_config.json"
CROP_SCALE = 0.9
IMAGE_SIZE = 224


def _load_openpi_history() -> types.ModuleType:
    path = REPO_ROOT / "policy_servers" / "openpi" / "finetune" / "history.py"
    spec = importlib.util.spec_from_file_location("openvlaoft_bench_history", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to locate {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


history = _load_openpi_history()


def prompt_text(task: str) -> str:
    return f"In: What action should the robot take to {task.lower()}?\nOut:"


def num_images(exterior_history: tuple[int, ...]) -> int:
    return 2 + len(exterior_history)


def _crop_size(size: int) -> int:
    return round(size * CROP_SCALE**0.5)


@dataclasses.dataclass(frozen=True)
class AugParams:
    top: int
    left: int
    brightness: float
    contrast: float
    saturation: float
    hue: float


def sample_aug(rng: np.random.Generator, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> AugParams:
    ch, cw = _crop_size(height), _crop_size(width)
    return AugParams(
        top=int(rng.integers(0, height - ch + 1)),
        left=int(rng.integers(0, width - cw + 1)),
        brightness=float(rng.uniform(-0.2, 0.2)),
        contrast=float(rng.uniform(0.8, 1.2)),
        saturation=float(rng.uniform(0.8, 1.2)),
        hue=float(rng.uniform(-0.05, 0.05)),
    )


def _crop_resize(x: torch.Tensor, top: int, left: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    return tvf.resized_crop(x, top, left, _crop_size(h), _crop_size(w), [h, w], antialias=False)


def _to_float(img: np.ndarray) -> torch.Tensor:
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) uint8 image, got {img.dtype} {img.shape}")
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0


def _to_uint8(x: torch.Tensor) -> np.ndarray:
    return (x.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()


def augment(img: np.ndarray, p: AugParams) -> np.ndarray:
    x = _crop_resize(_to_float(img), p.top, p.left).clamp(0, 1)
    x = (x + p.brightness).clamp(0, 1)
    mean = x.mean(dim=(-2, -1), keepdim=True)
    x = ((x - mean) * p.contrast + mean).clamp(0, 1)
    x = tvf.adjust_saturation(x, p.saturation).clamp(0, 1)
    x = tvf.adjust_hue(x, p.hue).clamp(0, 1)
    return _to_uint8(x)


def center_crop(img: np.ndarray) -> np.ndarray:
    x = _to_float(img)
    h, w = x.shape[-2:]
    return _to_uint8(_crop_resize(x, (h - _crop_size(h)) // 2, (w - _crop_size(w)) // 2))


class ArrayStats(TypedDict):
    mean: list[float]
    std: list[float]
    max: list[float]
    min: list[float]
    q01: list[float]
    q99: list[float]
    mask: list[bool]


class DatasetStats(TypedDict):
    action: ArrayStats
    proprio: ArrayStats
    num_transitions: int
    num_trajectories: int


def array_stats(x: np.ndarray) -> ArrayStats:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "max": x.max(0).tolist(),
        "min": x.min(0).tolist(),
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
        "mask": [True] * x.shape[1],
    }


def normalize(x: np.ndarray, stats: ArrayStats) -> np.ndarray:
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    out = np.clip(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1)
    out = np.where(np.asarray(stats["mask"]), out, x)
    return np.where(np.asarray(stats["min"]) == np.asarray(stats["max"]), 0.0, out).astype(np.float32)


def to_delta(actions: np.ndarray, joint_position: np.ndarray) -> np.ndarray:
    out = np.array(actions, dtype=np.float32)
    out[..., :ARM_DIM] -= joint_position[..., None, :ARM_DIM]
    return out


def to_absolute(actions: np.ndarray, joint_position: np.ndarray) -> np.ndarray:
    out = np.array(actions, dtype=np.float32)
    out[..., :ARM_DIM] += joint_position[..., None, :ARM_DIM]
    return out


def write_stats(run_dir: pathlib.Path, stats: DatasetStats) -> None:
    (run_dir / STATS_FILE).write_text(json.dumps({UNNORM_KEY: stats}, indent=2))


def read_stats(path: pathlib.Path) -> DatasetStats:
    data = json.loads(path.read_text())
    stats = data[UNNORM_KEY]
    for key, dim in (("action", ACTION_DIM), ("proprio", PROPRIO_DIM)):
        for field in ("q01", "q99", "min", "max", "mask"):
            if len(stats[key][field]) != dim:
                raise ValueError(f"{path}: {key}.{field} has {len(stats[key][field])} dims, expected {dim}")
    return stats
