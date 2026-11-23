from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import pathlib
from typing import TypedDict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from history import clamped_indices, validate_offsets
from PIL import Image
from pydantic import BaseModel, ConfigDict
from speed_conditioning import CENTER, SCALE, Normalisation, normalize

EXPECTED_SCHEMA = "droid_panda_v3"
PREPARED_META = "bench_prepared.json"
VIDEO_FRAMES = 8
ACTION_HORIZON = 7
IMAGE_SIZE = 224
VIDEO_SIZE = 256
ACTION_DIM = 8
STATE_DIM = 8

_EXT = "exterior_image_1_left"
_WRIST = "wrist_image_left"
_SCHEMA = pa.schema(
    [
        (_EXT, pa.binary()),
        (_WRIST, pa.binary()),
        ("state", pa.list_(pa.float32(), STATE_DIM)),
        ("actions", pa.list_(pa.float32(), ACTION_DIM)),
    ]
)


class EpisodeEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    episode_index: int
    chunk: int
    batch: int
    length: int
    task: str
    belt_speed: float | None = None


class PreparedMeta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str
    bench_schema: str
    fps: int
    action_min: list[float]
    action_max: list[float]
    episodes: list[EpisodeEntry]


class Sample(TypedDict):
    image: list[Image.Image]
    video: np.ndarray
    action: np.ndarray
    state: np.ndarray
    lang: str


def _info(src: pathlib.Path) -> dict:
    info = json.loads((src / "meta" / "info.json").read_text())
    stamp = info.get("bench_schema")
    if not isinstance(stamp, dict) or stamp.get("name") != EXPECTED_SCHEMA:
        raise RuntimeError(f"{src} declares camera schema {stamp!r}; the bench client renders {EXPECTED_SCHEMA!r}")
    return info


def _convert_chunk(args: tuple[str, str, int, list[int]]) -> tuple[int, list[int], np.ndarray, np.ndarray]:
    src, out, chunk, episodes = args
    lengths: list[int] = []
    lo = np.full(ACTION_DIM, np.inf, dtype=np.float64)
    hi = np.full(ACTION_DIM, -np.inf, dtype=np.float64)
    path = pathlib.Path(out) / f"chunk-{chunk:03d}.arrow"
    tmp = path.with_suffix(".arrow.tmp")
    with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_file(sink, _SCHEMA) as writer:
        for ep in episodes:
            table = pq.read_table(pathlib.Path(src) / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet")
            frame_index = np.asarray(table.column("frame_index").to_pylist())
            if not np.array_equal(frame_index, np.arange(len(frame_index))):
                raise ValueError(f"episode {ep}: frame_index is not 0..{len(frame_index) - 1}")
            joints = np.asarray(table.column("joint_position").to_pylist(), dtype=np.float32)
            gripper = np.asarray(table.column("gripper_position").to_pylist(), dtype=np.float32)
            actions = np.asarray(table.column("actions").to_pylist(), dtype=np.float32)
            lo = np.minimum(lo, actions.min(0))
            hi = np.maximum(hi, actions.max(0))
            batch = pa.record_batch(
                [
                    table.column(_EXT).combine_chunks().field("bytes"),
                    table.column(_WRIST).combine_chunks().field("bytes"),
                    pa.FixedSizeListArray.from_arrays(
                        pa.array(np.concatenate([joints, gripper], 1).ravel()), STATE_DIM
                    ),
                    pa.FixedSizeListArray.from_arrays(pa.array(actions.ravel()), ACTION_DIM),
                ],
                schema=_SCHEMA,
            )
            writer.write_batch(batch)
            lengths.append(len(frame_index))
    tmp.replace(path)
    return chunk, lengths, lo, hi


def load_belt_speeds(path: pathlib.Path) -> dict[int, float]:
    speeds = json.loads(path.read_text())
    if not isinstance(speeds, list) or not all(isinstance(v, (int, float)) for v in speeds):
        raise ValueError(f"{path}: expected a JSON list of numbers indexed by episode_index")
    return {i: float(v) for i, v in enumerate(speeds)}


def _check_against_provenance(src: pathlib.Path, speeds: dict[int, float], episodes: list[int]) -> None:
    path = src / "meta" / "bench_episode_provenance.jsonl"
    if not path.is_file():
        return
    recorded = {
        int(r["episode_index"]): float(r["belt_speed"]) for r in map(json.loads, path.read_text().splitlines()) if r
    }
    for ep in episodes:
        if speeds[ep] != recorded[ep]:
            raise ValueError(f"episode {ep}: belt speed list says {speeds[ep]}, {path} says {recorded[ep]}")


def prepare(
    src: pathlib.Path,
    out: pathlib.Path,
    *,
    max_episodes: int | None,
    workers: int,
    belt_speeds: dict[int, float] | None = None,
) -> PreparedMeta:
    info = _info(src)
    tasks = [json.loads(line) for line in (src / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    if max_episodes is not None:
        tasks = tasks[:max_episodes]
    if belt_speeds is not None:
        episodes = [int(r["episode_index"]) for r in tasks]
        missing = [ep for ep in episodes if ep not in belt_speeds]
        if missing:
            raise ValueError(f"no belt speed for {len(missing)} episodes, e.g. {missing[:5]}")
        _check_against_provenance(src, belt_speeds, episodes)
    chunk_size = int(info["chunks_size"])
    by_chunk: dict[int, list[int]] = {}
    for row in tasks:
        by_chunk.setdefault(int(row["episode_index"]) // chunk_size, []).append(int(row["episode_index"]))
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(src), str(out), chunk, eps) for chunk, eps in sorted(by_chunk.items())]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_convert_chunk, jobs)
    task_of = {int(r["episode_index"]): str(r["tasks"][0]) for r in tasks}
    entries: list[EpisodeEntry] = []
    lo = np.full(ACTION_DIM, np.inf)
    hi = np.full(ACTION_DIM, -np.inf)
    for (chunk, lengths, c_lo, c_hi), (_, _, _, eps) in zip(results, jobs, strict=True):
        lo, hi = np.minimum(lo, c_lo), np.maximum(hi, c_hi)
        for batch, (ep, length) in enumerate(zip(eps, lengths, strict=True)):
            entries.append(
                EpisodeEntry(
                    episode_index=ep,
                    chunk=chunk,
                    batch=batch,
                    length=length,
                    task=task_of[ep],
                    belt_speed=None if belt_speeds is None else belt_speeds[ep],
                )
            )
    meta = PreparedMeta(
        source=str(src),
        bench_schema=EXPECTED_SCHEMA,
        fps=int(info["fps"]),
        action_min=lo.tolist(),
        action_max=hi.tolist(),
        episodes=entries,
    )
    (out / PREPARED_META).write_text(meta.model_dump_json())
    return meta


def load_meta(root: pathlib.Path) -> PreparedMeta:
    return PreparedMeta.model_validate_json((root / PREPARED_META).read_text())


def normalize_actions(actions: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.where(hi > lo, hi - lo, 1.0)
    return np.where(hi > lo, 2.0 * (actions - lo) / span - 1.0, 0.0).astype(np.float32)


def unnormalize_actions(normalized: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (0.5 * (np.clip(normalized, -1.0, 1.0) + 1.0) * (hi - lo) + lo).astype(np.float32)


def _png(data: bytes, size: int) -> Image.Image:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return image if image.size == (size, size) else image.resize((size, size), Image.BILINEAR)


class BenchDataset:
    def __init__(self, root: pathlib.Path, history: tuple[int, ...], *, speed_conditioned: bool = False) -> None:
        self.root = root
        self.history = validate_offsets(history)
        self.meta = load_meta(root)
        self.speed_norm: Normalisation | None = {"center": CENTER, "scale": SCALE} if speed_conditioned else None
        if speed_conditioned and any(e.belt_speed is None for e in self.meta.episodes):
            raise ValueError(f"{root} was prepared without belt speeds; rerun bench_data.py with --belt-speeds")
        self.action_min = np.asarray(self.meta.action_min, dtype=np.float32)
        self.action_max = np.asarray(self.meta.action_max, dtype=np.float32)
        lengths = np.asarray([e.length for e in self.meta.episodes], dtype=np.int64)
        self._starts = np.concatenate([[0], np.cumsum(lengths)])
        self._readers: dict[int, pa.ipc.RecordBatchFileReader] = {}

    def __len__(self) -> int:
        return int(self._starts[-1])

    def locate(self, index: int) -> tuple[EpisodeEntry, int]:
        ep = int(np.searchsorted(self._starts, index, side="right") - 1)
        return self.meta.episodes[ep], index - int(self._starts[ep])

    def _batch(self, entry: EpisodeEntry) -> pa.RecordBatch:
        reader = self._readers.get(entry.chunk)
        if reader is None:
            source = pa.memory_map(str(self.root / f"chunk-{entry.chunk:03d}.arrow"), "r")
            reader = self._readers[entry.chunk] = pa.ipc.open_file(source)
        return reader.get_batch(entry.batch)

    def __getitem__(self, index: int) -> Sample:
        entry, t = self.locate(index)
        batch = self._batch(entry)
        ext, wrist = batch.column(_EXT), batch.column(_WRIST)
        last = entry.length - 1
        future = [min(last, t + k) for k in range(VIDEO_FRAMES)]
        video = np.stack(
            [np.stack([np.asarray(_png(col[i].as_py(), VIDEO_SIZE)) for i in future]) for col in (ext, wrist)]
        )
        images = [_png(ext[i].as_py(), IMAGE_SIZE) for i in clamped_indices(t, self.history)]
        images.append(_png(wrist[t].as_py(), IMAGE_SIZE))
        actions = np.asarray(
            [batch.column("actions")[min(last, t + k)].as_py() for k in range(ACTION_HORIZON)], dtype=np.float32
        )
        state = np.asarray([batch.column("state")[t].as_py()], dtype=np.float32)
        if self.speed_norm is not None:
            assert entry.belt_speed is not None
            state = np.concatenate([state, [[normalize(entry.belt_speed, self.speed_norm)]]], axis=1, dtype=np.float32)
        return {
            "image": images,
            "video": video,
            "action": normalize_actions(actions, self.action_min, self.action_max),
            "state": state,
            "lang": entry.task,
        }

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_readers": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a bench LeRobot dataset for VLA-JEPA training.")
    parser.add_argument("--src", required=True, type=pathlib.Path, help="local LeRobot v2.1 dataset root")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--belt-speeds", type=pathlib.Path, default=None, help="per-episode belt speeds (the speed-input variant)"
    )
    args = parser.parse_args()
    meta = prepare(
        args.src.expanduser(),
        args.out.expanduser(),
        max_episodes=args.max_episodes,
        workers=args.workers,
        belt_speeds=None if args.belt_speeds is None else load_belt_speeds(args.belt_speeds),
    )
    total = sum(e.length for e in meta.episodes)
    print(f"prepared {len(meta.episodes)} episodes / {total} frames into {args.out}")
    print(f"action min {np.round(meta.action_min, 4).tolist()}\naction max {np.round(meta.action_max, 4).tolist()}")


if __name__ == "__main__":
    main()
