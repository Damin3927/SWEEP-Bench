"""Write the assets the current-frame and past-frame runs share: normalisation stats and T5 prompt embeddings."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import pickle

import numpy as np

from .bench_dataset import STATS_FILE, T5_FILE, episode_path, read_episodes, read_info, read_numeric


def compute_stats(root: pathlib.Path, workers: int) -> dict[str, list[float]]:
    info = read_info(root)
    paths = [episode_path(root, info, e["episode_index"]) for e in read_episodes(root)]
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        episodes = list(pool.map(read_numeric, paths))
    stats: dict[str, list[float]] = {}
    for key in ("actions", "proprio"):
        values = np.concatenate([e[key] for e in episodes], axis=0)
        for name, fn in (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std), ("median", np.median)):
            stats[f"{key}_{name}"] = fn(values, axis=0).astype(np.float64).tolist()
    stats["num_frames"] = [float(sum(len(e["actions"]) for e in episodes))]
    return stats


def compute_t5(root: pathlib.Path) -> dict:
    from cosmos_policy.datasets.t5_embedding_utils import generate_t5_embeddings

    tasks = [json.loads(line)["task"] for line in (root / "meta" / "tasks.jsonl").read_text().splitlines() if line]
    return generate_t5_embeddings(sorted(set(tasks)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=pathlib.Path, required=True)
    parser.add_argument("--assets-dir", type=pathlib.Path, required=True)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--t5", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    if args.stats:
        stats = compute_stats(args.dataset_dir, args.workers)
        (args.assets_dir / STATS_FILE).write_text(json.dumps(stats, indent=2))
        print(f"wrote {args.assets_dir / STATS_FILE} over {int(stats['num_frames'][0])} frames", flush=True)
    if args.t5:
        embeddings = compute_t5(args.dataset_dir)
        with open(args.assets_dir / T5_FILE, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"wrote {args.assets_dir / T5_FILE} with {len(embeddings)} prompts", flush=True)


if __name__ == "__main__":
    main()
