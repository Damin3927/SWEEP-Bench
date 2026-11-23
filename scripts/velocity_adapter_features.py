"""Extract optical-flow features from a LeRobot pool for the belt-speed adapter."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from sweep_bench.velocity_adapter import ImageFlow

VIDEO_KEY = "observation.images.opst_cam"


def decode(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def read_images(path: Path, key: str) -> np.ndarray:
    cells = pq.read_table(path, columns=[key]).column(key).to_pylist()
    return np.stack([np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB")) for cell in cells])


def episode_frames(root: Path, info: dict, key: str, ep: int) -> np.ndarray:
    chunk = ep // info["chunks_size"]
    if info["features"][key]["dtype"] == "image":
        return read_images(root / info["data_path"].format(episode_chunk=chunk, episode_index=ep), key)
    return decode(root / info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=ep))


def claim_output(out: Path, config: dict) -> None:
    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise SystemExit(f"{out} holds features from a different extraction config; pass a new --out")
        return
    if any(out.glob("episode_*.npz")):
        raise SystemExit(f"{out} holds episode features with no config.json; pass a new --out")
    out.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=1))


def save_episode(target: Path, **arrays) -> None:
    partial = target.with_name(target.name + ".partial")
    try:
        with partial.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--speeds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-key", default=VIDEO_KEY)
    parser.add_argument("--flow-side", type=int, default=112, help="side of the pooled flow grid")
    parser.add_argument("--per-family", type=int, default=10)
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--gap", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    speeds = json.loads(args.speeds.read_text())
    info = json.loads((args.root / "meta" / "info.json").read_text())
    families = json.loads((args.root / "meta" / "ranges.json").read_text())
    rng = np.random.default_rng(args.seed)
    plan = []
    for family in families:
        first, last = family["episodes"]
        chosen = rng.choice(np.arange(first, last + 1), size=args.per_family, replace=False)
        plan += [(family["gym_id"], int(ep)) for ep in sorted(chosen)]

    flow = ImageFlow(out_side=args.flow_side)
    dt = args.gap / info["fps"]
    config = {k: v for k, v in vars(args).items() if k not in ("out", "threads")}
    config |= {"speeds_sha256": hashlib.sha256(args.speeds.read_bytes()).hexdigest(), "dt": dt}
    claim_output(args.out, json.loads(json.dumps(config, default=str)))

    for n, (gym_id, ep) in enumerate(plan):
        target = args.out / f"episode_{ep:06d}.npz"
        if target.exists():
            continue
        frames = episode_frames(args.root, info, args.image_key, ep)
        episode_rng = np.random.default_rng([args.seed, ep])
        starts = episode_rng.integers(args.gap, len(frames) - args.pairs + 1, size=args.windows)
        ends = (starts[:, None] + np.arange(args.pairs)[None, :]).reshape(-1)
        flows = flow(frames[ends - args.gap], frames[ends])
        save_episode(
            target,
            flow=flows.astype(np.float16).reshape(args.windows, args.pairs, *flows.shape[1:]),
            end_frames=ends.reshape(args.windows, args.pairs),
            speed=np.float32(speeds[ep]),
            gym_id=gym_id,
            episode=ep,
            length=len(frames),
        )
        if n % 20 == 0:
            print(f"{n + 1}/{len(plan)} episode {ep} {gym_id} speed {speeds[ep]:.3f}", flush=True)


if __name__ == "__main__":
    main()
