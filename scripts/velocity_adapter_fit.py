"""Fit the belt-speed head on optical-flow features and report accuracy against the number of training episodes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sweep_bench.velocity_adapter import SPEED_CENTER, SPEED_SCALE, PooledCNN, flow_cnn_inputs

SPEED_BINS = np.arange(0.10, 0.4001, 0.05)
HEAD = "flow_cnn"


def load(features: Path) -> dict[str, np.ndarray]:
    files = sorted(features.glob("episode_*.npz"))
    if not files:
        raise SystemExit(f"no episode_*.npz under {features}")
    cols = defaultdict(list)
    for f in files:
        z = np.load(f)
        if "flow" not in z:
            raise SystemExit(
                f"{f} holds no optical flow; extract the features with scripts/velocity_adapter_features.py"
            )
        cols["flow"].append(z["flow"].astype(np.float32))
        cols["speed"].append(float(z["speed"]))
        cols["gym_id"].append(str(z["gym_id"]))
        cols["episode"].append(int(z["episode"]))
    return {k: np.stack(v) if k == "flow" else np.asarray(v) for k, v in cols.items()}


def sub_windows(stored: np.ndarray, length: int) -> np.ndarray:
    e, w, p = stored.shape[:3]
    n_sub = p // length
    out = stored[:, :, : n_sub * length].reshape(e, w, n_sub, length, *stored.shape[3:]).mean(axis=3)
    return out.reshape(e * w * n_sub, *stored.shape[3:])


def labels(data: dict[str, np.ndarray], episodes: np.ndarray, per_episode: int) -> dict[str, np.ndarray]:
    return {k: np.repeat(data[k][episodes], per_episode) for k in ("speed", "episode", "gym_id")}


def windows(data: dict[str, np.ndarray], episodes: np.ndarray, length: int) -> dict[str, np.ndarray]:
    flow = sub_windows(data["flow"][episodes], length)
    return {"flow": flow, **labels(data, episodes, len(flow) // len(episodes))}


def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int, batch: int, seed: int) -> nn.Module:
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        idx = torch.randint(len(x), (min(batch, len(x)),), generator=gen)
        loss = nn.functional.smooth_l1_loss(model(x[idx]).reshape(-1), y[idx], beta=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def fit_predict(
    tr: dict,
    te_x: torch.Tensor,
    steps: int,
    seed: int,
    center: float = SPEED_CENTER,
    scale: float = SPEED_SCALE,
    save: Path | None = None,
) -> np.ndarray:
    torch.manual_seed(seed)
    x = flow_cnn_inputs(tr["flow"])
    y = torch.from_numpy(((tr["speed"] - center) / scale).astype(np.float32))
    model = train(PooledCNN(x.shape[1], stride=2), x, y, steps, 256, seed)
    if save is not None:
        torch.save(model.state_dict(), save)
    with torch.no_grad():
        s = model(te_x).reshape(-1).numpy()
    return s * scale + center


def speed_groups(speed: np.ndarray) -> dict[str, np.ndarray]:
    rounded = np.round(speed, 2)
    values = np.unique(rounded)
    if len(values) <= 8:
        return {f"{v:.2f}": rounded == v for v in values}
    return {f"{lo:.2f}": (speed >= lo) & (speed < lo + 0.05 + 1e-9) for lo in SPEED_BINS[:-1]}


def metrics(pred: np.ndarray, win: dict) -> dict:
    err = pred - win["speed"]
    by_bin = {}
    for key, m in speed_groups(win["speed"]).items():
        by_bin[key] = {"bias": float(err[m].mean()), "mae": float(np.abs(err[m]).mean()), "n": int(m.sum())}
    fam = defaultdict(list)
    for g, e in zip(win["gym_id"], np.abs(err), strict=True):
        fam[str(g)].append(e)
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "p90": float(np.quantile(np.abs(err), 0.9)),
        "within_0.025": float((np.abs(err) <= 0.025).mean()),
        "by_speed": by_bin,
        "by_family_mae": {k: float(np.mean(v)) for k, v in fam.items()},
    }


def held_out(data: dict[str, np.ndarray], per_group: int, by_speed: bool) -> np.ndarray:
    groups = (
        np.array([f"{g}@{v}" for g, v in zip(data["gym_id"], data["speed"], strict=True)])
        if by_speed
        else data["gym_id"]
    )
    rng = np.random.default_rng(0)
    test = []
    for g in sorted(set(groups)):
        idx = np.flatnonzero(groups == g)
        test += list(rng.choice(idx, size=min(per_group, len(idx)), replace=False))
    return np.array(sorted(test))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test-per-family", type=int, default=2)
    parser.add_argument("--test-by-speed", action="store_true", help="hold out --test-per-family per (family, speed)")
    parser.add_argument(
        "--speed-center", type=float, default=SPEED_CENTER, help="the head regresses (v - center) / scale"
    )
    parser.add_argument("--speed-scale", type=float, default=SPEED_SCALE)
    parser.add_argument("--sizes", type=int, nargs="+", default=[640])
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-dir", type=Path, help="save each fitted head's state_dict here")
    args = parser.parse_args()
    if args.save_dir is not None and (args.speed_center, args.speed_scale) != (SPEED_CENTER, SPEED_SCALE):
        parser.error("--save-dir: OnlineBeltSpeed decodes heads with SPEED_CENTER/SPEED_SCALE, so keep the defaults")
    torch.set_num_threads(args.threads)

    data = load(args.features)
    test = held_out(data, args.test_per_family, args.test_by_speed)
    pool = np.setdiff1d(np.arange(len(data["speed"])), test)
    print(f"{len(data['speed'])} episodes: {len(test)} test, {len(pool)} in the training pool")

    results = []
    for length in args.lengths:
        te = windows(data, test, length)
        te_x = flow_cnn_inputs(te["flow"])
        for size in [s for s in args.sizes if s <= len(pool)]:
            for draw in range(args.draws):
                chosen = np.random.default_rng([size, draw]).choice(pool, size=size, replace=False)
                tr = windows(data, chosen, length)
                save = None
                if args.save_dir is not None:
                    args.save_dir.mkdir(parents=True, exist_ok=True)
                    save = args.save_dir / f"{HEAD}_L{length}_X{size}_draw{draw}.pt"
                pred = fit_predict(tr, te_x, args.steps, draw, args.speed_center, args.speed_scale, save)
                m = metrics(pred, te)
                results.append({"length": length, "episodes": size, "draw": draw, "head": HEAD, **m})
                print(
                    f"L={length} X={size:4d} draw={draw} {HEAD:10s} mae={m['mae']:.4f} rmse={m['rmse']:.4f} "
                    f"p90={m['p90']:.4f} within.025={m['within_0.025']:.2f}",
                    flush=True,
                )
    args.out.write_text(
        json.dumps({"config": vars(args) | {"features": str(args.features)}, "results": results}, indent=1, default=str)
    )


if __name__ == "__main__":
    main()
