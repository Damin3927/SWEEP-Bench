import argparse
import io
import json
import multiprocessing
import os
import pathlib
import shutil
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from conv_state import (
    FAMILY_RANGES,
    clear_incomplete,
    family_ranges_path,
    mark_incomplete,
    parse_family_ranges,
)
from huggingface_hub import snapshot_download
from lerobot.common.datasets.compute_stats import get_feature_stats
from lerobot.common.datasets.lerobot_dataset import (
    HF_LEROBOT_HOME,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.common.datasets.utils import (
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    TASKS_PATH,
    append_jsonlines,
    load_jsonlines,
    write_info,
)
from PIL import Image

from sweep_bench.data.schemas import DROID_PANDA_SCHEMA
from sweep_bench.data.vintage import INFO_KEY, check_schema_name, read_stamp
from sweep_bench.kinematics import (
    EULER_RX_WRAPPED,
    PandaFK,
    euler_from_quat_wxyz,
)


def dom_features(h: int, w: int) -> dict:
    image = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.images.opst_cam": dict(image),
        "observation.images.wrist_cam": dict(image),
        "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }


def decode_image(cell: dict) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


EULER_MARKER = "bench_euler_convention.json"


def write_euler_marker(root: pathlib.Path, convention: str = EULER_RX_WRAPPED) -> None:
    path = root / "meta" / EULER_MARKER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"euler_convention": convention}, indent=2))


def fetch_family_ranges(repo_id: str) -> pathlib.Path | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    for filename in (f"meta/{FAMILY_RANGES}", FAMILY_RANGES):
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        except EntryNotFoundError:
            continue
        except RepositoryNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001 -- an unreachable hub is not a verdict
            raise RuntimeError(
                f"could not determine whether {repo_id} carries a {FAMILY_RANGES} family manifest "
                f"({type(exc).__name__}: {exc}). Refusing to convert: if it does, dropping it here "
                f"silently costs the destination its per-family validation holdout. Retry, or pass a "
                f"local --src that already holds the manifest."
            ) from exc
        print(f"  fetched {filename} from {repo_id} (not in the local cache)")
        return pathlib.Path(path)
    return None


def resolve_family_ranges(
    src_root: pathlib.Path, total_episodes: int, repo_id: str | None = None
) -> pathlib.Path | None:
    src_ranges = family_ranges_path(src_root)
    if src_ranges is None and repo_id is not None:
        src_ranges = fetch_family_ranges(repo_id)
    if src_ranges is None:
        return None
    try:
        ranges = parse_family_ranges(src_ranges.read_text(), total_episodes)
    except ValueError as exc:
        raise RuntimeError(
            f"{src_ranges} is not a partition of the {total_episodes} episodes to convert: {exc}. "
            f"Copying it would attribute episodes to the wrong families; regenerate it against the source."
        ) from exc
    print(f"  {FAMILY_RANGES}: {len(ranges)} families, will be carried into the destination's meta/")
    return src_ranges


def write_family_ranges(src_ranges: pathlib.Path, dst_root: pathlib.Path) -> None:
    dst = dst_root / "meta" / FAMILY_RANGES
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_ranges, dst)
    print(f"  carried {FAMILY_RANGES} forward into meta/")


def episode_task_json(task_str: str) -> str:
    return json.dumps({"task": "raw", "instructions": [task_str]})


class RangeSummary(NamedTuple):
    pos_min: np.ndarray
    pos_max: np.ndarray
    deltas: list[float]
    grippers: list[float]


def convert_range(args: argparse.Namespace, dst_root: pathlib.Path, episodes: range) -> RangeSummary:
    src_root = HF_LEROBOT_HOME / args.src
    src_stamp = read_stamp(src_root)
    src_meta = LeRobotDatasetMetadata(args.src)
    h, w, _ = src_meta.features["exterior_image_1_left"]["shape"]
    dst = LeRobotDataset.create(
        repo_id=args.dst,
        fps=src_meta.fps,
        root=dst_root,
        features=dom_features(h, w),
        use_videos=True,
        image_writer_threads=8,
    )
    if src_stamp is not None:
        dst.meta.info[INFO_KEY] = src_stamp
        write_info(dst.meta.info, dst.root)

    mark_incomplete(dst.root, source=args.src)

    fk = PandaFK()
    pos_min, pos_max = np.full(3, np.inf), np.full(3, -np.inf)
    deltas, grippers = [], []

    for done, ep in enumerate(episodes, start=1):
        frame = pd.read_parquet(src_root / src_meta.get_data_file_path(ep))
        task_idx = int(np.asarray(frame["task_index"].iloc[0]).item())
        task_json = episode_task_json(src_meta.tasks[task_idx])
        for _, row in frame.iterrows():
            qpos = np.asarray(row["joint_position"], dtype=np.float32)
            act = np.asarray(row["actions"], dtype=np.float32)
            state_pos, state_quat = fk.tcp_pose(qpos)
            act_pos, act_quat = fk.tcp_pose(act[:7])

            state_euler = euler_from_quat_wxyz(state_quat)
            act_euler = euler_from_quat_wxyz(act_quat)
            act_euler = state_euler + np.mod(act_euler - state_euler + np.pi, 2 * np.pi) - np.pi

            state = np.concatenate([state_pos, state_euler])
            action = np.concatenate([act_pos, act_euler, act[7:8]])
            dst.add_frame(
                {
                    "observation.images.opst_cam": decode_image(row["exterior_image_1_left"]),
                    "observation.images.wrist_cam": decode_image(row["wrist_image_left"]),
                    "observation.state": state.astype(np.float32),
                    "action": action.astype(np.float32),
                    "task": task_json,
                }
            )
            pos_min = np.minimum(pos_min, state_pos)
            pos_max = np.maximum(pos_max, state_pos)
            deltas.append(np.linalg.norm(act_pos - state_pos))
            grippers.append(act[7])
        dst.save_episode()
        if done % 20 == 0:
            print(f"[{dst_root.name}: {done}/{len(episodes)}] episodes converted", flush=True)
    return RangeSummary(pos_min, pos_max, deltas, grippers)


def merge_shards(shard_roots: list[pathlib.Path], dst_root: pathlib.Path) -> None:
    info = json.loads((shard_roots[0] / "meta" / "info.json").read_text())
    chunks = info["chunks_size"]
    video_keys = [key for key, ft in info["features"].items() if ft["dtype"] == "video"]
    tasks: dict[str, int] = {}
    episode = frames = 0
    for shard in shard_roots:
        shard_tasks = {row["task_index"]: row["task"] for row in load_jsonlines(shard / TASKS_PATH)}
        shard_stats = {row["episode_index"]: row["stats"] for row in load_jsonlines(shard / EPISODES_STATS_PATH)}
        for row in load_jsonlines(shard / EPISODES_PATH):
            local = row["episode_index"]
            data_path = info["data_path"].format(episode_chunk=local // chunks, episode_index=local)
            table = pq.read_table(shard / data_path)
            n = table.num_rows
            task_index = np.array(
                [tasks.setdefault(shard_tasks[t], len(tasks)) for t in table["task_index"].to_pylist()]
            )
            columns = {
                "episode_index": np.full((n,), episode),
                "index": np.arange(frames, frames + n),
                "task_index": task_index,
            }
            stats = shard_stats[local]
            for name, values in columns.items():
                i = table.schema.get_field_index(name)
                table = table.set_column(i, table.schema.field(i), pa.array(values, type=table.schema.field(i).type))
                stats[name] = {k: v.tolist() for k, v in get_feature_stats(values, axis=(0,), keepdims=True).items()}
            out = dst_root / info["data_path"].format(episode_chunk=episode // chunks, episode_index=episode)
            out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out)
            for key in video_keys:
                src_video = shard / info["video_path"].format(
                    episode_chunk=local // chunks, video_key=key, episode_index=local
                )
                dst_video = dst_root / info["video_path"].format(
                    episode_chunk=episode // chunks, video_key=key, episode_index=episode
                )
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                os.replace(src_video, dst_video)
            append_jsonlines({"episode_index": episode, "tasks": row["tasks"], "length": n}, dst_root / EPISODES_PATH)
            append_jsonlines({"episode_index": episode, "stats": stats}, dst_root / EPISODES_STATS_PATH)
            episode += 1
            frames += n
    for task, task_index in tasks.items():
        append_jsonlines({"task_index": task_index, "task": task}, dst_root / TASKS_PATH)
    info.update(
        total_episodes=episode,
        total_frames=frames,
        total_tasks=len(tasks),
        total_videos=episode * len(video_keys),
        total_chunks=(episode - 1) // chunks + 1,
        splits={"train": f"0:{episode}"},
    )
    write_info(info, dst_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="source DROID-schema repo id")
    parser.add_argument("--dst", required=True, help="destination DOM-schema repo id (local)")
    parser.add_argument("--workers", type=int, default=1, help="convert this many episode ranges in parallel")
    parser.add_argument("--force", action="store_true", help="overwrite existing destination")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error(f"--workers must be at least 1, got {args.workers}")

    src_root = HF_LEROBOT_HOME / args.src
    if not src_root.exists():
        snapshot_download(repo_id=args.src, repo_type="dataset", local_dir=src_root)
    check_schema_name(src_root, DROID_PANDA_SCHEMA.name)
    total = LeRobotDatasetMetadata(args.src).total_episodes
    workers = min(args.workers, total)

    family_ranges = resolve_family_ranges(
        src_root,
        total,
        repo_id=None if pathlib.Path(args.src).expanduser().is_absolute() else args.src,
    )

    dst_root = HF_LEROBOT_HOME / args.dst
    shards_root = dst_root.with_name(dst_root.name + ".shards")
    for path in (dst_root, shards_root) if workers > 1 else (dst_root,):
        if path.exists():
            if not args.force:
                raise FileExistsError(f"{path} exists; pass --force to overwrite")
            shutil.rmtree(path)

    if workers == 1:
        summaries = [convert_range(args, dst_root, range(total))]
    else:
        mark_incomplete(dst_root, source=args.src)
        bounds = np.linspace(0, total, workers + 1).astype(int)
        shard_roots = [shards_root / f"{k:03d}" for k in range(workers)]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            futures = [
                pool.submit(convert_range, args, root, range(lo, hi))
                for root, lo, hi in zip(shard_roots, bounds[:-1], bounds[1:], strict=True)
            ]
            summaries = [future.result() for future in futures]
        merge_shards(shard_roots, dst_root)
        shutil.rmtree(shards_root)

    if family_ranges is not None:
        write_family_ranges(family_ranges, dst_root)
    write_euler_marker(dst_root, EULER_RX_WRAPPED)
    clear_incomplete(dst_root)
    pos_min = np.min([s.pos_min for s in summaries], axis=0)
    pos_max = np.max([s.pos_max for s in summaries], axis=0)
    deltas = [d for s in summaries for d in s.deltas]
    grippers = [g for s in summaries for g in s.grippers]
    print(f"Done: {dst_root}")
    print(f"  state pos min/max: {np.round(pos_min, 3)} / {np.round(pos_max, 3)}")
    print(f"  |action_pos - state_pos|: mean {np.mean(deltas):.4f} m, p99 {np.quantile(deltas, 0.99):.4f} m")
    print(f"  action gripper range: [{np.min(grippers):.2f}, {np.max(grippers):.2f}]")


if __name__ == "__main__":
    main()
