"""Sweep a served policy across the evaluated belt speeds, one cell per speed (one cell per static task)."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import zlib

DEFAULT_SPEEDS: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
STATIC_PREFIX = "Static"
STATIC_LABEL = "static"

_REPO = pathlib.Path(__file__).resolve().parent.parent

CELL_KEYS = frozenset(
    {
        "family",
        "speed",
        "accel",
        "episodes",
        "success_rate",
        "mean_score",
        "seed",
        "policy",
        "exec_mode",
        "platform_style",
        "early_fail",
        "seconds",
        "background_pool_spec",
        "background_pool",
        "episode_backgrounds",
        "speed_token",
        "true_speeds",
        "told_speeds",
        "adapter_head",
        "fixed_tol_speed",
        "episode_records",
    }
)


def cell_seed(base: int, family: str, speed: float) -> int:
    return base + (zlib.crc32(family.encode()) % 100_000)


_RESOLVE_POOL = (
    "import json, sys; from sweep_bench.tasks.conveyor_base.placements import resolve_background_pool; "
    "print(json.dumps([list(b) for b in resolve_background_pool(sys.argv[1])]))"
)


def resolve_pool(spec: str, cmd: list[str], cwd: pathlib.Path, env: dict[str, str] | None = None) -> list[object]:
    proc = subprocess.run([*cmd, "-c", _RESOLVE_POOL, spec], cwd=cwd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"cannot resolve --background-pool {spec}: {(proc.stderr or proc.stdout)[-400:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def pool_record(stdout: str) -> dict[str, object]:
    record: dict[str, object] = {}
    for line in stdout.splitlines():
        for key in ("background_pool", "episode_backgrounds"):
            if line.startswith(f"{key}: "):
                record[key] = json.loads(line.removeprefix(f"{key}: "))
    return record


def episode_records(stdout: str) -> list[dict[str, object]]:
    return [json.loads(line.removeprefix("episode: ")) for line in stdout.splitlines() if line.startswith("episode: ")]


def is_static(family: str) -> bool:
    return family.startswith(STATIC_PREFIX)


def cell_label(speed: float, accel: float) -> str:
    return f"{speed:.2f}" if accel == 0.0 else f"{speed:.2f}_a{accel:+.4f}"


def family_cells(family: str, grid: list[tuple[float, float]]) -> list[tuple[float, float, str]]:
    if is_static(family):
        return [(0.0, 0.0, STATIC_LABEL)]
    return [(s, a, cell_label(s, a)) for s, a in grid]


def fixed_tol_speed() -> float | None:
    from sweep_bench.tasks.conveyor_base.on_belt import FIXED_TOL_SPEED_ENV

    value = os.environ.get(FIXED_TOL_SPEED_ENV)
    return float(value) if value else None


def run_cell(
    family: str, speed: float, args: argparse.Namespace, out_path: pathlib.Path, accel: float = 0.0
) -> tuple[dict[str, object] | None, str]:
    # fmt: off
    cmd = [
        "uv", "run", "--no-sync", "python", "scripts/eval.py",
        "--task", family,
        "--policy", args.policy,
        "--port", str(args.port),
        "--randomize",
        "--seed", str(cell_seed(args.seed, family, speed)),
        "--num-episodes", str(args.episodes),
        "--num-envs", "1",
        "--exec-mode", args.exec_mode,
    ]
    # fmt: on
    if not is_static(family):
        cmd += ["--speed-range", f"{speed}", f"{speed}"]
    platform_style = getattr(args, "platform_style", "box")
    cmd += ["--platform-style", platform_style]
    pool = getattr(args, "background_pool", None)
    cmd += ["--background-pool", "none" if pool is None else pool]
    speed_token = getattr(args, "speed_token", False)
    if speed_token:
        cmd.append("--speed-token")
    adapter_head = getattr(args, "adapter_head", None)
    if adapter_head is not None:
        cmd += ["--adapter-head", str(adapter_head)]
    if accel != 0.0:
        cmd += ["--accel", f"{accel}"]
    started = time.time()
    proc = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
    elapsed = time.time() - started

    rate = score = None
    speeds: dict[str, list[float]] = {}
    tolerance_speed: float | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("speed_token:"):
            speeds = json.loads(line.removeprefix("speed_token:"))
        if line.startswith("fixed_tol_speed:"):
            tolerance_speed = json.loads(line.removeprefix("fixed_tol_speed:"))
        if line.startswith("success_rate:"):
            parts = line.replace(":", " ").split()
            rate = float(parts[1])
            if "mean_score" in line:
                score = float(parts[parts.index("mean_score") + 1])
    if rate is None:
        tail = (proc.stderr or proc.stdout)[-800:]
        return None, f"no success_rate in output (rc={proc.returncode})\n{tail}"
    backgrounds = pool_record(proc.stdout) if pool is not None else {}
    episode_backgrounds = backgrounds.get("episode_backgrounds")
    if pool is not None and not (isinstance(episode_backgrounds, list) and len(episode_backgrounds) == args.episodes):
        return None, f"eval.py printed no background for every episode under --background-pool {pool}"
    expected = getattr(args, "background_pool_resolved", None)
    if pool is not None and expected is not None and backgrounds.get("background_pool") != expected:
        return None, f"eval.py drew from another pool than --background-pool {pool} resolved to when the sweep started"
    episodes = episode_records(proc.stdout)
    if episodes and (
        len(episodes) != args.episodes
        or f"{sum(bool(e['success']) for e in episodes) / len(episodes):.3f}" != f"{rate:.3f}"
    ):
        return None, "eval.py's per-episode lines disagree with its episode count or success rate"

    told_printed = (
        speeds.get("told") == "adapter" if adapter_head is not None else len(speeds.get("told", [])) == args.episodes
    )
    if speed_token and not (len(speeds.get("true", [])) == args.episodes and told_printed):
        return None, f"--speed-token cell printed no speed for every episode (rc={proc.returncode})"

    cell = {
        "family": family,
        "speed": speed,
        "accel": accel,
        "episodes": args.episodes,
        "success_rate": rate,
        "mean_score": score,
        "seed": cell_seed(args.seed, family, speed),
        "policy": args.policy,
        "exec_mode": args.exec_mode,
        "platform_style": platform_style,
        "early_fail": "on",
        "seconds": round(elapsed, 1),
    }
    if tolerance_speed is not None:
        cell["fixed_tol_speed"] = tolerance_speed
    if pool is not None:
        cell |= {"background_pool_spec": pool, **backgrounds}
    if speed_token:
        cell |= {
            "speed_token": True,
            "true_speeds": speeds["true"],
            "told_speeds": speeds["told"],
        }
    if adapter_head is not None:
        cell["adapter_head"] = str(adapter_head)
    if episodes:
        cell["episode_records"] = episodes
    out_path.write_text(json.dumps(cell, indent=2))
    return cell, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument(
        "--speeds",
        default=",".join(str(s) for s in DEFAULT_SPEEDS),
        help=f"belt speeds (m/s), comma-separated; a static task ({STATIC_PREFIX}*) runs one cell on its still belt",
    )
    ap.add_argument(
        "--accels",
        default="0",
        help="belt accelerations (m/s^2), comma-separated; each --speeds value is then the window-centre speed",
    )
    ap.add_argument("--episodes", type=int, default=20, help="episodes per (family, speed) cell")
    ap.add_argument("--seed", type=int, default=1000, help="base seed; the same value pairs two policies")
    ap.add_argument("--exec-mode", default="sync")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true", help="re-measure cells that are already on disk")
    ap.add_argument("--platform-style", default="table", choices=["box", "table"])
    ap.add_argument(
        "--background-pool",
        default="passing",
        help="eval.py's --background-pool: 'passing' (default), a JSON list file, or 'none'. Every cell of a "
        "family uses one seed, so every speed and every policy run with --seed sees the same backgrounds.",
    )
    ap.add_argument("--speed-token", action="store_true", help="pass eval.py --speed-token to every cell")
    ap.add_argument(
        "--adapter-head",
        type=pathlib.Path,
        default=None,
        help="pass eval.py --adapter-head PT to every cell: the velocity adapter's speed replaces the true one",
    )
    args = ap.parse_args()
    if args.background_pool == "none":
        args.background_pool = None
    args.fixed_tol_speed = fixed_tol_speed()
    speeds = [float(s) for s in args.speeds.split(",")]
    accels = [float(a) for a in args.accels.split(",")]
    grid = [(s, a) for s in speeds for a in accels]
    labels = [cell_label(s, a) for s, a in grid]
    if len(set(labels)) != len(labels):
        sys.exit(
            f"--speeds/--accels give two cells one file name ({sorted({x for x in labels if labels.count(x) > 1})}); "
            "speeds must differ at 2 decimals and accelerations at 4"
        )
    if args.adapter_head is not None:
        args.adapter_head = str(args.adapter_head.resolve())
    args.background_pool_resolved = (
        None
        if args.background_pool is None
        else resolve_pool(args.background_pool, ["uv", "run", "--no-sync", "python"], _REPO)
    )
    if args.platform_style == "table":
        from sweep_bench.assets.table_materials import cache_problems, default_cache_dir, load_manifest

        problems = [p for m in load_manifest().materials for p in cache_problems(m, default_cache_dir())]
        if problems:
            sys.exit(
                f"--platform-style table needs every table material in {default_cache_dir()}; "
                f"{len(problems)} map(s) are missing or wrong, e.g. {problems[0]}. "
                f"Run `python scripts/fetch_table_materials.py`."
            )

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, object]] = []
    failures: list[str] = []
    total = sum(len(family_cells(family, grid)) for family in args.families)
    done = 0
    for family in args.families:
        for speed, accel, label in family_cells(family, grid):
            done += 1
            path = out / f"{family}__{label}.json"
            if path.is_file() and not args.overwrite:
                cached = json.loads(path.read_text())
                if cached.get("platform_style", "box") != args.platform_style:
                    sys.exit(
                        f"{path} was measured with platform_style={cached.get('platform_style', 'box')!r}, "
                        f"this sweep asks for {args.platform_style!r}; use another --out or --overwrite"
                    )
                if (cached.get("background_pool_spec"), cached.get("background_pool")) != (
                    args.background_pool,
                    args.background_pool_resolved,
                ):
                    sys.exit(
                        f"{path} was measured with background_pool={cached.get('background_pool_spec')!r} "
                        f"({len(cached.get('background_pool') or [])} backgrounds), this sweep asks for "
                        f"{args.background_pool!r} ({len(args.background_pool_resolved or [])}); use another --out "
                        "or --overwrite"
                    )
                foreign = sorted(key for key, value in cached.items() if key not in CELL_KEYS and value is not None)
                if foreign:
                    sys.exit(
                        f"{path} carries {foreign}, which this sweep does not write, so it was measured under "
                        "another condition; use another --out or --overwrite"
                    )
                arm = (cached.get("speed_token", False), cached.get("adapter_head"))
                wanted = (args.speed_token, args.adapter_head)
                if arm != wanted:
                    sys.exit(
                        f"{path} was measured with (speed_token, adapter_head)={arm}, this sweep asks for "
                        f"{wanted}; use another --out or --overwrite"
                    )
                if cached.get("early_fail", "off") != "on":
                    sys.exit(
                        f"{path} was measured with early_fail={cached.get('early_fail', 'off')!r}, and every episode of "
                        "this sweep ends early once success can no longer happen; use another --out or --overwrite"
                    )
                if cached.get("fixed_tol_speed") != args.fixed_tol_speed:
                    sys.exit(
                        f"{path} was measured with fixed_tol_speed={cached.get('fixed_tol_speed')!r}, this sweep "
                        f"runs with {args.fixed_tol_speed!r}; use another --out or --overwrite"
                    )
                if cached.get("accel", 0.0) != accel:
                    sys.exit(f"{path} was measured at accel={cached.get('accel', 0.0)}, this sweep asks for {accel}")
                run = {
                    "episodes": args.episodes,
                    "seed": cell_seed(args.seed, family, speed),
                    "policy": args.policy,
                    "exec_mode": args.exec_mode,
                }
                measured = {key: cached.get(key) for key, value in run.items() if cached.get(key) != value}
                if measured:
                    wanted_run = {key: run[key] for key in measured}
                    sys.exit(
                        f"{path} was measured with {measured}, this sweep asks for {wanted_run}; "
                        "use another --out or --overwrite"
                    )
                cells.append({"accel": 0.0, **cached})
                print(f"[{done}/{total}] {family} @ {label}  (cached)", flush=True)
                continue
            cell, err = run_cell(family, speed, args, path, accel=accel)
            if cell is None:
                failures.append(f"{family} @ {label}: {err}")
                print(f"[{done}/{total}] {family} @ {label}  FAILED -- {err.splitlines()[0]}", flush=True)
                continue
            cells.append(cell)
            print(
                f"[{done}/{total}] {family} @ {label}  success={cell['success_rate']:.3f}  ({cell['seconds']:.0f}s)",
                flush=True,
            )

    summary = {
        "policy": args.policy,
        "episodes_per_cell": args.episodes,
        "speeds": speeds,
        "accels": accels,
        "families": args.families,
        "base_seed": args.seed,
        "platform_style": args.platform_style,
        "early_fail": "on",
        "fixed_tol_speed": args.fixed_tol_speed,
        "cells": cells,
        "failed_cells": failures,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'family':<30}" + "".join(f"{cell_label(s, a):>16}" for s, a in grid) + f"{'mean':>8}")
    by = {(c["family"], c["speed"], c["accel"]): c["success_rate"] for c in cells}
    for family in args.families:
        row = [by.get((family, s, a)) for s, a, _ in family_cells(family, grid)]
        got = [r for r in row if r is not None]
        cellstr = "".join(f"{r:>16.2f}" if r is not None else f"{'--':>16}" for r in row)
        mean = f"{sum(got) / len(got):>8.2f}" if got else f"{'--':>8}"
        print(f"{family.replace('Conveyor', '').replace('-v0', ''):<30}{cellstr}{mean}")

    if failures:
        print(f"\n{len(failures)} cell(s) produced no number:", file=sys.stderr)
        for f in failures:
            print(f"  {f.splitlines()[0]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
