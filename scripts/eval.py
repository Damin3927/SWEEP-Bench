"""Success-rate evaluation CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

from sweep_bench.assets.table_materials import cache_problems, default_cache_dir, load_manifest
from sweep_bench.evaluation import build_policy, evaluate, make_env
from sweep_bench.execution import all_executors
from sweep_bench.tasks.conveyor_base.on_belt import FIXED_TOL_SPEED_ENV
from sweep_bench.tasks.conveyor_base.placements import resolve_background_pool
from sweep_bench.tasks.registry import all_tasks, build_randomization_config, get_task, resolve_task_id

if TYPE_CHECKING:
    from collections.abc import Callable

    import gymnasium as gym
    import numpy as np

    from sweep_bench.policies import Obs, Policy
    from sweep_bench.policies.websocket import WebsocketPolicy
    from sweep_bench.velocity_adapter import OnlineBeltSpeed

_VLA_POLICIES = (
    "openpi",
    "openpi_ee",
    "dynamicvla",
    "fastwam",
    "vlajepa",
    "openvlaoft",
    "groot",
    "cosmospolicy",
)
_EE_POSE_POLICIES = ("openpi_ee", "dynamicvla", "fastwam")
_SPEED_TOKEN_POLICIES = ("openpi", "dynamicvla", "fastwam", "vlajepa")
_EE_POSE_CONTROL_MODE = "pd_ee_pose"


def adapter_speed_source(
    adapter: OnlineBeltSpeed, policy: WebsocketPolicy, composite: Callable[[Obs], np.ndarray]
) -> Callable[[Obs], float]:
    def speed(obs: Obs) -> float:
        had_estimate = bool(adapter.pair_estimates)
        value = adapter(composite(obs))
        if not had_estimate and adapter.pair_estimates:
            policy.replan()
        return value

    return speed


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
        help="Evaluate on the default task's static counterpart (the belt does not move).",
    )
    parser.add_argument("--robot", default="panda_wristcam")
    parser.add_argument("--policy", choices=["random", "zero", *_VLA_POLICIES], default="random")
    parser.add_argument("--host", default="localhost", help="websocket host for served policies")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="defaults to 1 for VLA policies, 16 otherwise",
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument(
        "--episode-start",
        type=int,
        default=0,
        help="score episodes N .. N+num_episodes-1 of the seeded draw (the same episodes as that slice of a longer "
        "run with the same seed); single env only",
    )
    parser.add_argument(
        "--exec-mode",
        default="sync",
        choices=sorted(all_executors()),
        help="Execution model. 'sync' ignores inference time (the simulation pauses while the policy "
        "predicts, every step); 'async' is serial (infer between chunks, arm holds); 'concurrent' "
        "pipelines inference on a background thread and plays each chunk whole (lags the world "
        "by the inference time); 'concurrent-laas' adds latency-aware action streaming (discards "
        "the stale prefix and overwrites with the newest chunk; openpi/dynamicvla only). "
        "'async'/'concurrent[-laas]' are single-env and need a chunk-inference policy.",
    )
    parser.add_argument(
        "--inference-latency",
        type=float,
        default=None,
        help="For async/concurrent: model inference as taking this many SECONDS, deterministically "
        "(round(latency/control_dt) sim steps), instead of measuring real wall-clock. Reproducible "
        "and host-independent — for eval/ablation. Omit to use the measured + real-time-paced path.",
    )
    parser.add_argument("--control-mode", default=None)
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Enable per-episode domain randomization. Mirrors scripts/collect.py so eval matches training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the episode RNG, so the run replays the same episodes. Pass the SAME seed to every "
        "cell of a sweep and to every policy you compare, or the cells are different populations.",
    )
    parser.add_argument(
        "--speed-range",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=None,
        help="Override the belt-speed randomization range (m/s). Pass the SAME value twice to pin a "
        "speed and sweep it -- `--speed-range 0.25 0.25` -- which is how a per-speed cell is measured "
        "with every other knob still randomized. Requires --randomize.",
    )
    parser.add_argument(
        "--accel",
        type=float,
        default=None,
        help="Constant belt acceleration (m/s^2, along travel) inside the window centred on the robot; "
        "the --speed-range draw becomes the speed at the window centre. Requires --randomize.",
    )
    parser.add_argument(
        "--accel-window",
        type=float,
        default=None,
        help="Length (m) of the acceleration window. Omit to keep the config default.",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Record each episode's third-person rollout to an mp4 in this dir. Single-env VLA eval only.",
    )
    parser.add_argument(
        "--scene-style",
        default=None,
        choices=["table", "replicacad", "ai2thor", "robocasa"],
        help="Background scene style with --background-pool none. Omit to keep the task's own default.",
    )
    parser.add_argument(
        "--scene-idx",
        type=int,
        default=0,
        help="Background build-config index for the non-'table' scene styles, with --background-pool none.",
    )
    parser.add_argument(
        "--platform-style",
        default="table",
        choices=["box", "table"],
        help="What the robot and the belt stand on: the table with a CC0 surface material, drawn per scene "
        "rebuild under --randomize (default, the evaluation protocol; needs scripts/fetch_table_materials.py "
        "once), or the plain box.",
    )
    parser.add_argument(
        "--background-pool",
        default="passing",
        help="Rebuild the scene on a background drawn from a pool at every episode end: 'passing' (default, the "
        "evaluation protocol: every background that passed the background check), a JSON file holding a list "
        "of [scene_style, scene_idx], or 'none' to use --scene-style/--scene-idx. The sequence is a function "
        "of --seed, so every policy and speed run with one seed sees the same backgrounds. Single env. Prints "
        "the pool and every episode's background.",
    )
    parser.add_argument(
        "--record-contact",
        action="store_true",
        help="Record, per episode, the first control step the gripper touches the task's object and the step the "
        "TCP comes nearest it, each with the belt speed and the object's along-travel position then. Off by "
        "default. Each episode line carries contact_* and closest_*.",
    )
    parser.add_argument(
        "--speed-token",
        action="store_true",
        help="Send the episode's commanded primary-belt speed (m/s) as `belt_speed` in every request, for a "
        f"speed-conditioned checkpoint. Single env; --policy in {_SPEED_TOKEN_POLICIES}.",
    )
    parser.add_argument(
        "--adapter-head",
        type=Path,
        default=None,
        metavar="PT",
        help="With --speed-token: send the speed the velocity adapter reads from the exterior composite instead of "
        "the true one: the mean of the latest 5 single-pair `flow_cnn` estimates (state_dict PT), 0.25 m/s until "
        "the first, with a new request as soon as the first exists. RAFT runs on the client GPU. --exec-mode sync.",
    )
    args = parser.parse_args()
    if args.speed_token and args.policy not in _SPEED_TOKEN_POLICIES:
        parser.error(f"--speed-token needs --policy in {_SPEED_TOKEN_POLICIES}")
    if args.adapter_head is not None and not args.speed_token:
        parser.error("--adapter-head replaces the speed --speed-token sends; pass --speed-token too")
    if args.adapter_head is not None and args.exec_mode != "sync":
        parser.error("--adapter-head reads every control step's frame in predict, which only --exec-mode sync calls")
    background_pool = None if args.background_pool == "none" else resolve_background_pool(args.background_pool)
    if background_pool is not None and (args.scene_style is not None or args.scene_idx != 0):
        parser.error("--scene-style/--scene-idx pick one background; pass --background-pool none with them")
    if args.platform_style == "table":
        problems = [p for m in load_manifest().materials for p in cache_problems(m, default_cache_dir())]
        if problems:
            parser.error(
                f"--platform-style table needs every table material in {default_cache_dir()}, and "
                f"{len(problems)} map(s) are missing or wrong, e.g. {problems[0]}. "
                f"Run `python scripts/fetch_table_materials.py`."
            )

    task_id = resolve_task_id(args.task, args.static)
    env_cls = get_task(task_id).env_cls

    rcfg = build_randomization_config(env_cls, args.randomize) if args.randomize else None

    if args.speed_range is not None:
        if rcfg is None or not getattr(rcfg, "speed", False):
            parser.error(
                "--speed-range needs the belt-speed knob on, which --randomize turns on; "
                "without it the override is inert and every cell runs at the task default"
            )
        if not hasattr(rcfg, "speed_range"):
            parser.error(f"{task_id} has no speed_range to override")
        rcfg.speed_range = (float(args.speed_range[0]), float(args.speed_range[1]))

    if args.accel is not None or args.accel_window is not None:
        if rcfg is None or not hasattr(rcfg, "accel"):
            parser.error("--accel/--accel-window need --randomize on a moving-belt task")
        if args.accel is not None:
            rcfg.accel = float(args.accel)
        if args.accel_window is not None:
            rcfg.accel_window = float(args.accel_window)

    default_envs = 1 if (args.policy in _VLA_POLICIES or background_pool is not None) else 16
    num_envs: int = args.num_envs if args.num_envs is not None else default_envs
    if args.video_dir and num_envs != 1:
        parser.error("--video-dir records one env per episode; pass --num-envs 1")
    if args.speed_token and num_envs != 1:
        parser.error("--speed-token sends env 0's belt speed; pass --num-envs 1")
    if background_pool is not None and num_envs != 1:
        parser.error(
            "--background-pool rebuilds the scene at every episode end; pass --num-envs 1 or --background-pool none"
        )

    control_mode = args.control_mode
    if args.policy in _EE_POSE_POLICIES:
        if control_mode not in (None, _EE_POSE_CONTROL_MODE):
            parser.error(
                f"--policy {args.policy} emits ABSOLUTE end-effector poses, so it requires "
                f"--control-mode {_EE_POSE_CONTROL_MODE} (got {control_mode!r}). Any other controller "
                f"reads the same 7 floats as deltas or joint targets: nothing raises, and the number "
                f"the run reports is scored against an action the policy never asked for."
            )
        control_mode = _EE_POSE_CONTROL_MODE

    fruits = env_cls.SUPPORTED_FRUITS if (args.randomize and env_cls.SUPPORTED_FRUITS) else None
    objects = tuple(spec.id for spec in env_cls.OBJECT_CATALOG) if (args.randomize and env_cls.OBJECT_CATALOG) else None

    on_episode_start = None
    follow_prompt = args.policy in _VLA_POLICIES and args.prompt is None
    true_speeds: list[float] = []
    adapter = None
    adapter_records: list[dict] = []
    if args.adapter_head is not None:
        import torch

        from sweep_bench.velocity_adapter import OnlineBeltSpeed

        if not torch.cuda.is_available():
            parser.error("--adapter-head runs RAFT on the client GPU, and none is visible")
        adapter = OnlineBeltSpeed(args.adapter_head, "cuda")
    if follow_prompt or args.speed_token:
        from sweep_bench.policies.episode_speed import episode_belt_speed
        from sweep_bench.policies.prompts import episode_prompt

        def on_episode_start(policy: Policy, env: gym.Env, _name: str = args.policy) -> None:
            if follow_prompt:
                p = episode_prompt(_name, env)
                if p is not None:
                    cast("WebsocketPolicy", policy).prompt = p
            if args.speed_token:
                if adapter is not None:
                    if adapter.sent:
                        adapter_records.append(adapter.summary(true_speeds[-1]))
                    adapter.reset()
                true = episode_belt_speed(env)
                true_speeds.append(true)
                if adapter is None:
                    cast("WebsocketPolicy", policy).belt_speed = true

    probe_env = make_env(
        task_id,
        args.robot,
        num_envs,
        prompt=args.prompt,
        control_mode=control_mode,
        randomization=rcfg,
        fruits=fruits,
        objects=objects,
        scene_style=args.scene_style,
        scene_idx=args.scene_idx,
        platform_style=args.platform_style,
        background_pool=background_pool,
    )
    prompt = getattr(probe_env.unwrapped, "prompt", "")
    policy = build_policy(args.policy, probe_env, num_envs, args.host, args.port, prompt)
    probe_env.close()
    conditioned = getattr(policy, "server_speed_conditioned", None)
    if args.speed_token and conditioned is not True:
        parser.error(f"--speed-token needs a server that declares speed_conditioned=True, got {conditioned}")
    if not args.speed_token and conditioned is True:
        parser.error("the server's checkpoint is speed-conditioned; evaluate it with --speed-token")
    if adapter is not None:
        from sweep_bench.policies.websocket import WebsocketPolicy
        from sweep_bench.velocity_adapter import exterior_composite

        if not isinstance(policy, WebsocketPolicy):
            parser.error(f"--adapter-head needs a served policy, got {type(policy).__name__}")
        policy.speed_source = adapter_speed_source(adapter, policy, exterior_composite)

    metrics = evaluate(
        policy,
        task_id=task_id,
        robot=args.robot,
        num_envs=num_envs,
        num_episodes=args.num_episodes,
        exec_mode=args.exec_mode,
        inference_latency=args.inference_latency,
        prompt=args.prompt,
        control_mode=control_mode,
        randomization=rcfg,
        fruits=fruits,
        objects=objects,
        scene_style=args.scene_style,
        scene_idx=args.scene_idx,
        platform_style=args.platform_style,
        background_pool=background_pool,
        on_episode_start=on_episode_start,
        video_dir=args.video_dir,
        seed=args.seed,
        record_contact=args.record_contact,
        episode_start=args.episode_start,
    )
    print(f"success_rate: {metrics['success_rate']:.3f}  ({args.num_episodes} eps, policy={args.policy})")
    print(f"record_contact: {'on' if metrics['record_contact'] else 'off'}")
    tolerance_speed = os.environ.get(FIXED_TOL_SPEED_ENV)
    print(f"fixed_tol_speed: {json.dumps(float(tolerance_speed) if tolerance_speed else None)}")
    if args.episode_start:
        print(f"episode_start: {args.episode_start}")
    if args.speed_token:
        true = true_speeds[: args.num_episodes]
        print(f"speed_token: {json.dumps({'true': true, 'told': true if adapter is None else 'adapter'})}")
    if adapter is not None and len(adapter_records) != len(metrics["episodes"]):
        raise SystemExit(f"{len(adapter_records)} adapter traces for {len(metrics['episodes'])} episodes")
    if background_pool is not None:
        print(f"background_pool: {json.dumps([list(b) for b in background_pool])}")
        print(f"episode_backgrounds: {json.dumps([list(b) for b in metrics['episode_backgrounds']])}")
    for i, episode in enumerate(metrics["episodes"]):
        extra = dict(adapter_records[i]) if adapter is not None else {}
        print(f"episode: {json.dumps({**episode, 'background': list(episode['background']), **extra})}")
    lat = metrics["mean_inference_latency_s"]
    if lat is not None:
        hold_frac = metrics["hold_fraction"]
        hold_frac_str = f"{hold_frac:.3f}" if hold_frac is not None else "n/a"
        print(
            f"  exec={args.exec_mode}  inference_latency: {lat * 1000:.0f} ms/call  "
            f"hold_steps/ep: {metrics['mean_hold_steps']:.1f}  hold_fraction: {hold_frac_str}  "
            f"inferences/ep: {metrics['mean_inferences_per_episode']:.1f}"
        )


if __name__ == "__main__":
    main()
