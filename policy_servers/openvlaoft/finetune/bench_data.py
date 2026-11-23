from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, TypedDict

import bench_common as bc
import datasets
import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Iterator

    from prismatic.vla.datasets.datasets import RLDSBatchTransform

NUMERIC_COLUMNS = ("episode_index", "frame_index", "task_index", "joint_position", "gripper_position", "actions")
EXTERIOR = "exterior_image_1_left"
WRIST = "wrist_image_left"


class Sample(TypedDict):
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    labels: torch.Tensor
    actions: np.ndarray
    proprio: np.ndarray
    dataset_name: str


def _tensor(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected a tensor from RLDSBatchTransform, got {type(value).__name__}")
    return value


class BenchFrames:
    def __init__(self, root: pathlib.Path, *, max_episodes: int | None = None) -> None:
        self.root = root
        info = json.loads((root / "meta" / "info.json").read_text())
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"{root}: expected a LeRobot v2.1 dataset, got {info.get('codebase_version')!r}")
        self.schema = info.get("bench_schema", {}).get("name")
        if self.schema != bc.DATASET_SCHEMA:
            raise ValueError(
                f"{root} declares camera schema {self.schema!r}; the bench client renders {bc.DATASET_SCHEMA!r}"
            )
        self.fps = int(info["fps"])
        episodes = sorted(
            (json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line),
            key=lambda e: e["episode_index"],
        )
        if max_episodes is not None:
            episodes = episodes[:max_episodes]
        files = [
            str(
                root
                / info["data_path"].format(
                    episode_chunk=e["episode_index"] // info["chunks_size"], episode_index=e["episode_index"]
                )
            )
            for e in episodes
        ]
        hf = datasets.load_dataset("parquet", data_files=files, split="train")
        if not isinstance(hf, datasets.Dataset):
            raise TypeError(f"expected a map-style datasets.Dataset, got {type(hf).__name__}")
        self.hf = hf
        self._image_columns = {c: self.hf.select_columns([c]) for c in (EXTERIOR, WRIST)}
        cols = self.hf.select_columns(list(NUMERIC_COLUMNS)).with_format("numpy")[:]
        self.joint_position = np.asarray(cols["joint_position"], dtype=np.float32)
        self.gripper_position = np.asarray(cols["gripper_position"], dtype=np.float32)
        self.actions = np.asarray(cols["actions"], dtype=np.float32)
        self.task_index = np.asarray(cols["task_index"], dtype=np.int64)
        episode_index = np.asarray(cols["episode_index"], dtype=np.int64)
        frame_index = np.asarray(cols["frame_index"], dtype=np.int64)
        self.num_episodes = len(episodes)

        lengths = np.array([e["length"] for e in episodes])
        ends = np.cumsum(lengths)
        starts = ends - lengths
        expected_episode = np.repeat([e["episode_index"] for e in episodes], lengths)
        if len(episode_index) != ends[-1] or not np.array_equal(episode_index, expected_episode):
            raise ValueError(f"{root}: rows are not the episodes of meta/episodes.jsonl in order")
        if not np.array_equal(frame_index, np.arange(len(frame_index)) - np.repeat(starts, lengths)):
            raise ValueError(f"{root}: frame_index is not 0..length-1 within each episode")
        self.start = np.repeat(starts, lengths)
        self.end = np.repeat(ends, lengths)
        tasks = (json.loads(line) for line in (root / "meta" / "tasks.jsonl").read_text().splitlines() if line)
        self.tasks = {int(t["task_index"]): str(t["task"]) for t in tasks}

    def __len__(self) -> int:
        return len(self.actions)

    def chunk_rows(self, rows: np.ndarray) -> np.ndarray:
        return np.minimum(rows[:, None] + np.arange(bc.NUM_ACTIONS_CHUNK), self.end[rows, None] - 1)

    def history_rows(self, row: int, offsets: tuple[int, ...]) -> list[int]:
        return [max(int(self.start[row]), row + o) for o in offsets]

    def proprio(self, rows: np.ndarray) -> np.ndarray:
        return np.concatenate([self.joint_position[rows], self.gripper_position[rows]], axis=-1)

    def delta_chunks(self, rows: np.ndarray) -> np.ndarray:
        return bc.to_delta(self.actions[self.chunk_rows(rows)], self.joint_position[rows])

    def images(self, rows: list[int], column: str) -> list[np.ndarray]:
        return [np.asarray(img.convert("RGB")) for img in self._image_columns[column][rows][column]]

    def stats(self, block: int = 65536) -> bc.DatasetStats:
        rows = np.arange(len(self))
        chunks = np.concatenate([self.delta_chunks(rows[i : i + block]) for i in range(0, len(rows), block)])
        return {
            "action": bc.array_stats(chunks.reshape(-1, bc.ACTION_DIM)),
            "proprio": bc.array_stats(self.proprio(rows)),
            "num_transitions": len(self),
            "num_trajectories": self.num_episodes,
        }


@dataclasses.dataclass
class BenchSampleDataset(torch.utils.data.Dataset["Sample"]):
    frames: BenchFrames
    stats: bc.DatasetStats
    batch_transform: RLDSBatchTransform
    image_transform: Callable[[Image.Image], torch.Tensor]
    exterior_history: tuple[int, ...]
    seed: int
    image_aug: bool = True

    _perms: dict[int, np.ndarray] = dataclasses.field(default_factory=dict, init=False, repr=False)

    def _permutation(self, epoch: int) -> np.ndarray:
        if epoch not in self._perms:
            self._perms.clear()
            self._perms[epoch] = np.random.default_rng([self.seed, epoch]).permutation(len(self.frames))
        return self._perms[epoch]

    def row(self, position: int) -> int:
        epoch, k = divmod(position, len(self.frames))
        return int(self._permutation(epoch)[k])

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Sample:
        position = index
        f = self.frames
        row = self.row(position)
        rows = np.array([row])
        exterior = f.images([*f.history_rows(row, self.exterior_history), row], EXTERIOR)
        wrist = f.images([row], WRIST)[0]
        if self.image_aug:
            rng = np.random.default_rng([self.seed, position, 1])
            shared = bc.sample_aug(rng)
            exterior = [bc.augment(img, shared) for img in exterior]
            wrist = bc.augment(wrist, bc.sample_aug(rng))
        rlds_batch: dict[str, object] = {
            "dataset_name": bc.UNNORM_KEY,
            "action": bc.normalize(f.delta_chunks(rows)[0], self.stats["action"]),
            "observation": {
                "image_primary": exterior[-1][None],
                "image_wrist": wrist[None],
                "proprio": bc.normalize(f.proprio(rows)[0], self.stats["proprio"]),
            },
            "task": {"language_instruction": f.tasks[int(f.task_index[row])].encode()},
        }
        out = self.batch_transform(rlds_batch)
        past = [self.image_transform(Image.fromarray(img)) for img in exterior[:-1]]
        images = [_tensor(out["pixel_values"]), _tensor(out["pixel_values_wrist"]), *past]
        return {
            "pixel_values": torch.cat(images, dim=0),
            "input_ids": _tensor(out["input_ids"]),
            "labels": _tensor(out["labels"]),
            "actions": np.asarray(out["actions"], dtype=np.float32),
            "proprio": np.asarray(out["proprio"], dtype=np.float32),
            "dataset_name": str(out["dataset_name"]),
        }


class StreamSampler(torch.utils.data.Sampler[int]):
    def __init__(self, start: int, stop: int, batch: int = 1, rank: int = 0, world: int = 1) -> None:
        if world > 1 and (start % batch or stop % batch):
            raise ValueError(f"start {start} and stop {stop} must be multiples of the micro-batch {batch}")
        self.start, self.stop, self.batch, self.rank, self.world = start, stop, batch, rank, world

    def _micro(self) -> range:
        return range(self.start // self.batch + self.rank, self.stop // self.batch, self.world)

    def __iter__(self) -> Iterator[int]:
        if self.world == 1:
            return iter(range(self.start, self.stop))
        return (m * self.batch + i for m in self._micro() for i in range(self.batch))

    def __len__(self) -> int:
        if self.world == 1:
            return self.stop - self.start
        return len(self._micro()) * self.batch
