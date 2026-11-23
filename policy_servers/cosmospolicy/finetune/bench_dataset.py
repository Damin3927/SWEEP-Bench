from __future__ import annotations

import functools
import io
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq
import torch
from cosmos_policy.datasets.dataset_common import compute_monte_carlo_returns, get_action_chunk_with_padding
from cosmos_policy.datasets.dataset_utils import preprocess_image
from PIL import Image

from .history import LatentLayout, clamped_frame_indices

PRIMARY_KEY = "exterior_image_1_left"
WRIST_KEY = "wrist_image_left"
_NUMERIC_COLUMNS = ("joint_position", "gripper_position", "actions")
STATS_FILE = "dataset_statistics.json"
T5_FILE = "t5_embeddings.pkl"


def read_info(root: pathlib.Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def bench_schema_name(root: pathlib.Path) -> str | None:
    schema = read_info(root).get("bench_schema")
    return schema.get("name") if isinstance(schema, dict) else None


def read_episodes(root: pathlib.Path, max_episodes: int | None = None) -> list[dict]:
    episodes = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    episodes.sort(key=lambda e: e["episode_index"])
    return episodes[:max_episodes] if max_episodes else episodes


def episode_path(root: pathlib.Path, info: dict, episode_index: int) -> pathlib.Path:
    chunk = episode_index // info["chunks_size"]
    return root / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)


def read_numeric(path: pathlib.Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path, columns=[*_NUMERIC_COLUMNS, "task_index"])
    out = {name: np.asarray(table[name].to_pylist(), dtype=np.float32) for name in _NUMERIC_COLUMNS}
    out["proprio"] = np.concatenate([out.pop("joint_position"), out.pop("gripper_position")], axis=1)
    out["task_index"] = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
    return out


def load_stats(assets_dir: pathlib.Path) -> dict[str, np.ndarray]:
    raw = json.loads((assets_dir / STATS_FILE).read_text())
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
    for key in ("actions", "proprio"):
        span = stats[f"{key}_max"] - stats[f"{key}_min"]
        if np.any(span <= 0):
            raise ValueError(f"{assets_dir / STATS_FILE}: {key} has a constant dimension {np.flatnonzero(span <= 0)}")
    return stats


def rescale(arr: np.ndarray, stats: dict[str, np.ndarray], key: str) -> np.ndarray:
    lo, hi = stats[f"{key}_min"], stats[f"{key}_max"]
    return (2 * (arr - lo) / (hi - lo) - 1).astype(np.float32)


def _decode(cell: dict) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


class BenchLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        assets_dir: str,
        history: tuple[int, ...] = (),
        chunk_size: int = 16,
        final_image_size: int = 224,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        gamma: float = 0.99,
        max_episodes: int | None = None,
    ) -> None:
        import pickle

        self.root = pathlib.Path(data_dir)
        self.assets_dir = pathlib.Path(assets_dir)
        self.layout = LatentLayout(tuple(history))
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.gamma = gamma
        self.info = read_info(self.root)
        self.episodes = read_episodes(self.root, max_episodes)
        self.tasks = {
            int(t["task_index"]): t["task"]
            for t in map(json.loads, (self.root / "meta" / "tasks.jsonl").read_text().splitlines())
        }
        self.stats = load_stats(self.assets_dir)
        with open(self.assets_dir / T5_FILE, "rb") as f:
            self.t5_text_embeddings: dict[str, torch.Tensor] = pickle.load(f)
        missing = sorted(set(self.tasks.values()) - set(self.t5_text_embeddings))
        if missing:
            raise ValueError(f"{self.assets_dir / T5_FILE} lacks {len(missing)} task prompts, e.g. {missing[:3]}")
        lengths = np.array([e["length"] for e in self.episodes], dtype=np.int64)
        self._starts = np.concatenate([[0], np.cumsum(lengths)])
        self.num_steps = int(self._starts[-1])

    def __len__(self) -> int:
        return self.num_steps

    def locate(self, idx: int) -> tuple[int, int]:
        ep = int(np.searchsorted(self._starts, idx, side="right") - 1)
        return ep, idx - int(self._starts[ep])

    @functools.lru_cache(maxsize=4)  # noqa: B019 (one dataset per worker; bounded)
    def _episode(self, ep: int) -> tuple[pq.ParquetFile, dict[str, np.ndarray]]:
        path = episode_path(self.root, self.info, self.episodes[ep]["episode_index"])
        return pq.ParquetFile(path), read_numeric(path)

    def _frames(self, ep: int, key: str, indices: tuple[int, ...]) -> dict[int, np.ndarray]:
        table = self._episode(ep)[0].read(columns=[key])[key]
        return {i: _decode(table[i].as_py()) for i in set(indices)}

    def __getitem__(self, idx: int) -> dict:
        ep, t = self.locate(idx)
        numeric = self._episode(ep)[1]
        num_steps = len(numeric["actions"])
        future = min(t + self.chunk_size, num_steps - 1)
        exterior_indices = clamped_frame_indices(t, self.layout.history)
        primary = self._frames(ep, PRIMARY_KEY, (*exterior_indices, future))
        wrist = self._frames(ep, WRIST_KEY, (t, future))

        source_images = {
            "blank": np.zeros_like(primary[t]),
            "wrist": wrist[t],
            "future_wrist": wrist[future],
            "future_exterior": primary[future],
            **{f"exterior@{o}": primary[i] for o, i in zip((*self.layout.history, 0), exterior_indices, strict=True)},
        }
        images = np.stack([source_images[s] for s in self.layout.frame_sources()])
        video = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=False,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )

        actions = rescale(numeric["actions"], self.stats, "actions")
        proprio = rescale(numeric["proprio"], self.stats, "proprio")
        returns = compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=self.gamma)
        next_t = min(t + self.chunk_size, num_steps - 1)
        next_future = min(next_t + self.chunk_size, num_steps - 1)
        task = self.tasks[int(numeric["task_index"][t])]
        layout = self.layout
        return {
            "video": video,
            "actions": get_action_chunk_with_padding(actions, t, self.chunk_size, num_steps),
            "t5_text_embeddings": torch.squeeze(self.t5_text_embeddings[task]),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": proprio[t],
            "future_proprio": proprio[future],
            "__key__": idx,
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "action_latent_idx": layout.action,
            "value_latent_idx": layout.value,
            "current_proprio_latent_idx": layout.current_proprio,
            "current_wrist_image_latent_idx": layout.current_wrist,
            "current_image_latent_idx": layout.current_exterior,
            "future_proprio_latent_idx": layout.future_proprio,
            "future_wrist_image_latent_idx": layout.future_wrist,
            "future_image_latent_idx": layout.future_exterior,
            "value_function_return": returns[future],
            "next_action_chunk": get_action_chunk_with_padding(actions, next_t, self.chunk_size, num_steps),
            "next_value_function_return": returns[next_future],
        }
