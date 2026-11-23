"""Collect scripted-grasp trajectories into a LeRobot dataset."""

from __future__ import annotations

import argparse

from sweep_bench.collection import (
    build_collection_env,
    check_env_matches_schema,
    collect_episodes,
    collection_stamp,
    push_to_hub,
)
from sweep_bench.data import LeRobotEpisodeRecorder
from sweep_bench.data.schemas import get_schema
from sweep_bench.tasks.conveyor_base.placements import resolve_background_pool
from sweep_bench.tasks.registry import all_tasks, resolve_task_id

_SEED_MAX = 2**32 - 1


def _seed(value: str) -> int:
    seed = int(value)
    if not 0 <= seed <= _SEED_MAX:
        raise argparse.ArgumentTypeError(
            f"seed must be between 0 and {_SEED_MAX} -- numpy.random.RandomState's range, which "
            f"ManiSkill hands the root to unchanged; got {seed}"
        )
    return seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default=None,
        choices=sorted(all_tasks()),
        help="Env id. Defaults to StaticAppleGrasp-v0 when --static is set, else ConveyorAppleGrasp-v0.",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Collect on the default task's static counterpart (the belt does not move and the object "
        "rests at a reachable point on it). --randomize then randomizes object position + orientation.",
    )
    parser.add_argument("--robot", default="panda_wristcam")
    parser.add_argument("--scene-style", default="replicacad", choices=["table", "replicacad", "ai2thor", "robocasa"])
    parser.add_argument(
        "--platform-style",
        default="table",
        choices=["box", "table"],
        help="What the robot and the belt stand on in the non-'table' scene styles: a table with a CC0 surface "
        "material (default, as the released demonstrations; drawn per scene rebuild under --randomize; needs "
        "scripts/fetch_table_materials.py once), or the plain box.",
    )
    parser.add_argument(
        "--scene-idx",
        type=int,
        default=0,
        help="Background build-config index for the non-'table' scene styles "
        "(replicacad apartment / ai2thor ArchitecTHOR scene / robocasa layout-style combo).",
    )
    parser.add_argument(
        "--background-pool",
        default="passing",
        help="Draw the background per scene rebuild from a pool: 'passing' (default, as the released "
        "demonstrations: every background that passed the background check, data/background_placements.json), "
        "a JSON file holding a list of [scene_style, scene_idx], or 'none' to use --scene-style/--scene-idx. "
        "The draw has its own stream, so every other draw is that of a fixed-background run with the same "
        "--seed.",
    )
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo_id (also the on-disk subdir).")
    parser.add_argument("--root", default=None, help="Override dataset root dir (default: HF_LEROBOT_HOME/<repo-id>).")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Parallel envs (GPU sim only). Linear speedup; macOS / CPU sim must keep this at 1.",
    )
    parser.add_argument("--max-steps", type=int, default=300, help="Hard cap per episode.")
    parser.add_argument("--schema", default="droid_panda_v3")
    parser.add_argument("--fps", type=int, default=20, help="Should match env.control_freq.")
    parser.add_argument(
        "--filter-success",
        action="store_true",
        help="Only save episodes where info['success'] is True at termination.",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Enable per-episode domain randomization (every boolean knob on the task's RandomizationConfig, "
        "plus the scripted velocity SDE; on fruit-drop tasks also the fruit identity).",
    )
    parser.add_argument(
        "--reconfigure-every",
        type=int,
        default=32,
        help="With --randomize: rebuild the scene every N saved episodes, which redraws the background, the "
        "object and the scene appearance. In-flight episodes are dropped at each rebuild.",
    )
    parser.add_argument(
        "--speed-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help="Override the conveyor speed randomization range (m/s), e.g. --speed-range 0.10 0.20. "
        "Requires --randomize. Default: the task's own speed_range.",
    )
    parser.add_argument(
        "--speed-bins",
        type=int,
        default=1,
        help="Stratify the SAVED episodes over K equal-width belt-speed bins with equal quotas, so "
        "--filter-success cannot skew the speed distribution toward slow belts (fast episodes fail "
        "more often). Requires --randomize; num-episodes must divide by K.",
    )
    parser.add_argument(
        "--accel-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help="Draw each episode's along-travel belt acceleration (m/s^2) uniformly from [LO, HI]. The drawn "
        "belt speed becomes the speed at the centre of the --accel-window; pin it with --speed-range V V.",
    )
    parser.add_argument(
        "--accel-window",
        type=float,
        default=None,
        help="Length (m) of the acceleration window centred on the robot base. Default: the task's own (1.0).",
    )
    parser.add_argument(
        "--accel-bins",
        type=int,
        default=1,
        help="Stratify the SAVED episodes over K equal-width bins of --accel-range with equal quotas, as "
        "--speed-bins does for speed. num-episodes must divide by K; cannot be combined with --speed-bins.",
    )
    parser.add_argument(
        "--seed",
        type=_seed,
        default=None,
        help="Seed the env's main RNG, which every episode draw descends from. UNSET (the default) "
        "leaves ManiSkill's fixed 2022+env_index, so two runs of one task at one --num-envs collect "
        "BIT-IDENTICAL episodes -- pass a different seed per process to split one task's episodes "
        "across several, or the shards duplicate each other instead of extending the dataset. The "
        "seed reaches the dataset as the stamp's main_seed, so independence is checkable after the "
        "fact from the artifacts alone.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Wipe the dataset dir if it exists.")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="After collection, push the dataset to the Hugging Face Hub as the dataset repo <repo-id>. "
        "Requires `huggingface-cli login` (or HF_TOKEN env var) with write scope.",
    )
    parser.add_argument(
        "--push-private",
        action="store_true",
        help="With --push-to-hub, create the HF repo as private (default: public).",
    )
    args = parser.parse_args()
    if args.accel_bins < 1:
        parser.error(f"--accel-bins must be at least 1, got {args.accel_bins}")
    if args.accel_bins > 1:
        if args.speed_bins > 1:
            parser.error("--accel-bins and --speed-bins cannot be combined")
        if not args.accel_range:
            parser.error("--accel-bins needs --accel-range")
        if args.num_episodes % args.accel_bins:
            parser.error(f"--num-episodes {args.num_episodes} must divide by --accel-bins {args.accel_bins}")

    schema = get_schema(args.schema)
    task_id = resolve_task_id(args.task, args.static)

    env = build_collection_env(
        task_id,
        args.robot,
        args.scene_style,
        "default",
        args.randomize,
        num_envs=args.num_envs,
        speed_range=tuple(args.speed_range) if args.speed_range else None,
        scene_idx=args.scene_idx,
        seed=args.seed,
        platform_style=args.platform_style,
        background_pool=None if args.background_pool == "none" else resolve_background_pool(args.background_pool),
        accel_range=tuple(args.accel_range) if args.accel_range else None,
        accel_window=args.accel_window,
    )
    check_env_matches_schema(env, schema)

    recorder = LeRobotEpisodeRecorder(
        repo_id=args.repo_id,
        schema=schema,
        fps=args.fps,
        root=args.root,
        collection=collection_stamp(
            env,
            task_id,
            args.robot,
            args.scene_style,
            args.scene_idx,
            "default",
            randomize=args.randomize,
            max_steps=args.max_steps,
            filter_success=args.filter_success,
            reconfigure_every=args.reconfigure_every,
            speed_bins=args.speed_bins,
            accel_bins=args.accel_bins,
        ),
        overwrite=args.overwrite,
    )

    saved, attempted = collect_episodes(
        env,
        recorder,
        schema,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        filter_success=args.filter_success,
        reconfigure_every=args.reconfigure_every,
        speed_bins=args.speed_bins,
        accel_bins=args.accel_bins,
    )

    dataset_root = recorder.root
    env.close()
    recorder.close()
    print(f"done: saved {saved} episode(s) over {attempted} attempt(s).")

    if args.push_to_hub:
        push_to_hub(args.repo_id, dataset_root, private=args.push_private)


if __name__ == "__main__":
    main()
