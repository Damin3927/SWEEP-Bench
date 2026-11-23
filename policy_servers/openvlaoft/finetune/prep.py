"""Build the dataset's arrow cache and its action / proprio statistics once."""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bench_common as bc  # noqa: E402
import bench_data  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True, help="directory for dataset_statistics.json")
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    t0 = time.time()
    frames = bench_data.BenchFrames(args.dataset_root, max_episodes=args.max_episodes)
    logging.info("%d frames / %d episodes loaded in %.0fs", len(frames), frames.num_episodes, time.time() - t0)
    stats = frames.stats()
    args.out.mkdir(parents=True, exist_ok=True)
    bc.write_stats(args.out, stats)
    logging.info("wrote %s in %.0fs total", args.out / bc.STATS_FILE, time.time() - t0)
    for key in ("action", "proprio"):
        logging.info("%s q01 %s", key, [round(v, 4) for v in stats[key]["q01"]])
        logging.info("%s q99 %s", key, [round(v, 4) for v in stats[key]["q99"]])


if __name__ == "__main__":
    main()
