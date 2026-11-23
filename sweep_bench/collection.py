from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

import gymnasium as gym
import numpy as np
import torch

from sweep_bench.assets.vetted import require_servable_catalogs
from sweep_bench.data import (
    build_frame,
    check_schema_describes_robot,
    extract_dataset_action,
    ik_fallback_detected,
)
from sweep_bench.data.provenance import CollectionStamp, EpisodeProvenance, config_snapshot
from sweep_bench.tasks.conveyor_base.randomization_scene import scene_randomization_active
from sweep_bench.tasks.registry import build_randomization_config, get_task

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from sweep_bench.data import LeRobotEpisodeRecorder
    from sweep_bench.data.provenance import DrawScalar, DrawValue
    from sweep_bench.data.schemas import BenchSchema
    from sweep_bench.tasks.conveyor_base.config import PlatformStyle, SceneStyle, VelocityRandomizationConfig
    from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
    from sweep_bench.tasks.conveyor_base.placements import Background
    from sweep_bench.tasks.conveyor_base.scripted import ScriptedControlMixin


_COLLECTION_ROBOTS = ("panda_wristcam",)


class CollectionProfileKwargs(TypedDict):
    velocity_randomization: VelocityRandomizationConfig
    fruits: NotRequired[tuple[str, ...]]
    objects: NotRequired[tuple[str, ...]]


class SceneIndexKwargs(TypedDict, total=False):
    replicacad_scene_idx: int
    ai2thor_scene_idx: int
    robocasa_scene_idx: int


class _PoolKwargs(TypedDict, total=False):
    background_pool: tuple[Background, ...]


def _scene_index_kwargs(scene_style: SceneStyle, scene_idx: int) -> SceneIndexKwargs:
    if scene_style == "replicacad":
        return {"replicacad_scene_idx": scene_idx}
    if scene_style == "ai2thor":
        return {"ai2thor_scene_idx": scene_idx}
    if scene_style == "robocasa":
        return {"robocasa_scene_idx": scene_idx}
    return {}


def collection_profile_kwargs(env_cls: type[ConveyorWorkspaceEnv], *, randomize: bool) -> CollectionProfileKwargs:
    kwargs: CollectionProfileKwargs = {"velocity_randomization": env_cls.VelocityRandomizationConfig(enabled=randomize)}
    if randomize and env_cls.SUPPORTED_FRUITS:
        kwargs["fruits"] = env_cls.SUPPORTED_FRUITS
    if randomize and env_cls.OBJECT_CATALOG:
        require_servable_catalogs(env_cls)
        kwargs["objects"] = tuple(spec.id for spec in env_cls.OBJECT_CATALOG)
    return kwargs


def check_env_matches_schema(env: gym.Env, schema: BenchSchema) -> None:
    check_schema_describes_robot(cast("ScriptedControlMixin", env.unwrapped), schema)


def build_collection_env(
    task_id: str,
    robot: str,
    scene_style: SceneStyle,
    shader_pack: str,
    randomize: bool,
    num_envs: int,
    speed_range: tuple[float, float] | None = None,
    scene_idx: int = 0,
    seed: int | None = None,
    platform_style: PlatformStyle = "box",
    background_pool: Sequence[Background] | None = None,
    accel_range: tuple[float, float] | None = None,
    accel_window: float | None = None,
) -> gym.Env:
    if robot not in _COLLECTION_ROBOTS:
        raise ValueError(
            f"the collection schema supports panda_wristcam only: robot={robot!r} is not one of "
            f"{list(_COLLECTION_ROBOTS)}. Plain 'panda' is excluded because it ships no "
            f"wrist camera and every frame carries a wrist stream; another arm additionally "
            f"means teaching data/schemas.py an arm_dof and a gripper normalization per robot."
        )
    env_cls = get_task(task_id).env_cls
    rcfg = build_randomization_config(env_cls, randomize)
    if speed_range is not None:
        if not hasattr(rcfg, "speed_range"):
            raise ValueError(f"{task_id} has no speed_range to override")
        rcfg.speed_range = (float(speed_range[0]), float(speed_range[1]))
    if accel_range is not None or accel_window is not None:
        if not hasattr(rcfg, "accel_range"):
            raise ValueError(f"{task_id} has no belt acceleration profile")
        if accel_range is not None:
            rcfg.accel_range = (float(accel_range[0]), float(accel_range[1]))
        if accel_window is not None:
            rcfg.accel_window = float(accel_window)
    profile_kwargs: CollectionProfileKwargs = collection_profile_kwargs(env_cls, randomize=randomize)
    scene_kwargs = _scene_index_kwargs(scene_style, scene_idx)
    pool_kwargs: _PoolKwargs = {} if background_pool is None else {"background_pool": tuple(background_pool)}
    env = gym.make(
        task_id,
        robot_uids=robot,
        obs_mode="rgbd",
        num_envs=num_envs,
        control_mode="pd_ee_delta_pose",
        scene_style=scene_style,
        platform_style=platform_style,
        shader_pack=shader_pack,
        randomization=rcfg,
        **profile_kwargs,
        **scene_kwargs,
        **pool_kwargs,
    )
    if seed is not None:
        env.reset(seed=seed, options={"reconfigure": True})
    return env


_NOT_A_DRAW = frozenset({"_elapsed_steps", "_episode_seed"})

_LATCHED_DRAW_MAX_WIDTH = 8


def collection_stamp(
    env: gym.Env,
    task_id: str,
    robot: str,
    scene_style: SceneStyle,
    scene_idx: int,
    shader_pack: str,
    *,
    randomize: bool,
    max_steps: int,
    filter_success: bool,
    reconfigure_every: int,
    speed_bins: int,
    accel_bins: int = 1,
) -> CollectionStamp:
    task = cast("ScriptedControlMixin", env.unwrapped)
    pool = tuple(getattr(task, "fruits", ())) or task.objects
    backgrounds = task.background_pool
    return CollectionStamp(
        task_id=task_id,
        robot=robot,
        scene_style="pool" if backgrounds is not None else scene_style,
        scene_idx=0 if backgrounds is not None else scene_idx,
        workspace_offset=None if backgrounds is not None else [float(v) for v in task.workspace_offset],
        shader_pack=shader_pack,
        control_mode=str(task.control_mode),
        num_envs=int(task.num_envs),
        randomize=randomize,
        object_pool=[str(o) for o in pool],
        main_seed=[int(s) for s in task._main_seed],
        max_steps=max_steps,
        filter_success=filter_success,
        reconfigure_every=reconfigure_every,
        speed_bins=speed_bins,
        accel_bins=accel_bins,
        expert_speed_scale=None,
        velrand_floor_ramp=[
            float(task._velrand_floor_top_effective()),
            float(task._scripted_velrand_floor_v_lo),
            float(task._scripted_velrand_floor_v_hi),
        ],
        platform_style=str(task.platform_style),
        background_pool=None if backgrounds is None else [[style, idx] for style, idx in backgrounds],
    )


def _draw_scalar(value: torch.Tensor | np.ndarray, name: str) -> DrawScalar:
    scalar = value.item()
    if isinstance(scalar, (bool, int, float, str)):
        return scalar
    raise TypeError(
        f"per-env draw {name!r} has dtype {value.dtype}, whose elements are {type(scalar).__name__} and not "
        f"provenance. Add it to `_NOT_A_DRAW` if it is not a draw, or teach this function its type."
    )


def _latched_draws(task: ScriptedControlMixin, env_index: int) -> dict[str, DrawValue]:
    draws: dict[str, DrawValue] = {}
    for name, value in sorted(vars(task).items()):
        if name in _NOT_A_DRAW or not isinstance(value, (torch.Tensor, np.ndarray)):
            continue
        shape = tuple(value.shape)
        if not shape or shape[0] != task.num_envs or len(shape) > 2:
            continue
        if len(shape) == 2 and shape[1] > _LATCHED_DRAW_MAX_WIDTH:
            continue
        row = value[env_index]
        draws[name] = _draw_scalar(row, name) if len(shape) == 1 else [_draw_scalar(v, name) for v in row]
    return draws


def episode_provenance(
    task: ScriptedControlMixin,
    env_index: int,
    *,
    reset_index: int,
    reconfigure_index: int,
    prompt: str,
) -> EpisodeProvenance:
    belt_velocities_for_env = [float(belt.velocity[env_index].item()) for belt in task._velocity_belts]
    identity = getattr(task, "episode_objects", None) or getattr(task, "episode_fruits", None)
    poses: dict[str, list[float]] = {}
    for actor in task._task_actors:
        pose = actor.pose
        row = [float(v) for v in pose.p[env_index].tolist()] + [float(v) for v in pose.q[env_index].tolist()]
        if actor.name in poses:
            raise ValueError(
                f"two task actors share the name {actor.name!r}, so their initial poses cannot be told apart"
            )
        poses[actor.name] = row
    return EpisodeProvenance(
        env_index=env_index,
        reset_index=reset_index,
        reconfigure_index=reconfigure_index,
        env_seed=int(task._episode_seed[env_index]),
        draw_seed=int(task._episode_seed[0]),
        prompt=prompt,
        object_id=None if identity is None else str(identity[env_index]),
        belt_speed=abs(belt_velocities_for_env[0]),
        belt_direction="-y" if belt_velocities_for_env[0] < 0.0 else "+y",
        belt_velocities=belt_velocities_for_env,
        randomization=config_snapshot(task.randomization),
        velocity_randomization=config_snapshot(task.velocity_randomization),
        initial_poses=poses,
        latched_draws=_latched_draws(task, env_index),
        belt_accel=float(task._belt_accel[env_index].item()) if task._belt_accel_on else None,
        belt_center_speed=float(task._belt_center_speed[env_index].item()) if task._belt_accel_on else None,
        scene_style=task.background[0],
        scene_idx=task.background[1],
        workspace_offset=[float(v) for v in task.workspace_offset],
    )


@dataclass(frozen=True)
class _SpeedBinState:
    index: int
    low: float
    high: float
    count: int
    quota: int
    label: str = "speed"
    unit: str = "m/s"


_IK_DEAD_RUN_ATTEMPTS = 8


_STALL_REF_KEEP_RATE = 0.05
_STALL_FALSE_ABORT_BUDGET = 1e-4
_STALL_MIN_ATTEMPTS = 200
_STALL_ATTEMPTS_PER_ENV = 25


def _stall_attempt_limit(num_envs: int, num_episodes: int) -> int:
    statistical = math.ceil(
        (math.log(num_episodes) + math.log(1.0 / _STALL_FALSE_ABORT_BUDGET)) / -math.log1p(-_STALL_REF_KEEP_RATE)
    )
    return max(_STALL_MIN_ATTEMPTS, _STALL_ATTEMPTS_PER_ENV * num_envs, statistical)


def _check_collection_stall(
    *,
    attempts_since_saved: int,
    num_envs: int,
    saved: int,
    num_episodes: int,
    bins: Sequence[_SpeedBinState] | None,
) -> None:
    limit = _stall_attempt_limit(num_envs, num_episodes)
    if attempts_since_saved <= limit:
        return
    lines = [
        f"collection stalled: {attempts_since_saved} consecutive episode(s) discarded with nothing saved "
        f"(limit {limit} for num_envs={num_envs}, num_episodes={num_episodes}); "
        f"{saved}/{num_episodes} episode(s) saved so far.",
    ]
    if bins is None:
        lines.append(
            "Every attempt was discarded, so the expert is not solving this config at all "
            "(--filter-success keeps successes only). Re-check the task and --speed-range before re-running."
        )
    else:
        label, unit = bins[0].label, bins[0].unit
        lines.append(f"Per-bin fill state (--{label}-bins aims every reset at the most-deficient bin):")
        lines += [
            f"  bin {b.index}: {b.count}/{b.quota} saved, {label} [{b.low:.4f}, {b.high:.4f}] {unit}" for b in bins
        ]
        lines.append(
            f"A bin whose {label} range the expert cannot solve pins every reset there forever. "
            f"Narrow --{label}-range to a band the expert succeeds on, or lower --{label}-bins."
        )
    raise RuntimeError("\n".join(lines))


def collect_episodes(
    env: gym.Env,
    recorder: LeRobotEpisodeRecorder,
    schema: BenchSchema,
    *,
    num_episodes: int,
    max_steps: int,
    filter_success: bool,
    reconfigure_every: int,
    speed_bins: int = 1,
    accel_bins: int = 1,
) -> tuple[int, int]:
    task = cast("ScriptedControlMixin", env.unwrapped)
    check_schema_describes_robot(task, schema)
    n = task.num_envs
    multi_fruit = len(getattr(task, "fruits", ())) > 1
    multi_object = len(getattr(task, "objects", ())) > 1
    scene_dr = scene_randomization_active(task.randomization)
    multi_background = task.background_pool is not None and len(task.background_pool) > 1
    needs_periodic_reconfigure = multi_fruit or multi_object or scene_dr or multi_background

    if speed_bins > 1 and accel_bins > 1:
        raise ValueError("stratify over speed or over acceleration, not both")
    stratify_accel = accel_bins > 1
    n_bins = accel_bins if stratify_accel else speed_bins
    stratify = n_bins > 1
    if stratify_accel:
        accel_range = task.randomization.accel_range
        if tuple(accel_range) == (0.0, 0.0):
            raise ValueError("accel_bins > 1 needs a drawn belt acceleration (--accel-range)")
        edges = np.linspace(*accel_range, accel_bins + 1)
    elif stratify:
        if not hasattr(task.randomization, "speed_range") or not task.randomization.speed:
            raise ValueError("speed_bins > 1 needs a task with randomized belt speed (--randomize)")
        edges = np.linspace(*task.randomization.speed_range, speed_bins + 1)
    if stratify:
        quota, rem = divmod(num_episodes, n_bins)
        if rem:
            raise ValueError(
                f"num_episodes={num_episodes} not divisible by {'accel' if stratify_accel else 'speed'}_bins={n_bins}"
            )
        bin_counts = [0] * n_bins

    def _stratum(i: int) -> int:
        drawn = pending_provenance[i]
        if stratify_accel:
            if drawn is None or drawn.belt_accel is None:
                raise RuntimeError(f"env {i} ended an episode with no drawn acceleration on record")
            value = drawn.belt_accel
        elif drawn is not None and drawn.belt_center_speed is not None:
            value = drawn.belt_center_speed
        else:
            value = abs(float(task.belt.drive.velocity[i].item()))
        return int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, n_bins - 1))

    def _episode_prompts() -> list[str]:
        prompt = str(getattr(task, "prompt", ""))
        return [str(p) for p in getattr(task, "episode_prompts", [prompt] * n)]

    pending_frames: list[list[dict]] = [[] for _ in range(n)]
    pending_steps = [0] * n
    pending_provenance: list[EpisodeProvenance | None] = [None] * n
    reset_index = 0
    reconfigure_index = 0
    saved = 0
    attempted = 0
    attempts_since_saved = 0
    last_reconfigure_saved = 0
    ik_failed = [False] * n
    ik_discarded = 0

    def _capture(env_indices: list[int]) -> None:
        for i in env_indices:
            pending_provenance[i] = episode_provenance(
                task,
                i,
                reset_index=reset_index,
                reconfigure_index=reconfigure_index,
                prompt=prompts[i],
            )

    obs, info = env.reset()
    task.reset_scripted()
    prompts = _episode_prompts()
    _capture(list(range(n)))

    while saved < num_episodes:
        ee_action = task.compute_scripted_action()
        next_obs, _reward, terminated, truncated, info = env.step(ee_action)

        for i in range(n):
            dataset_action = extract_dataset_action(task, ee_action, env_idx=i)
            if ik_fallback_detected(task, ee_action, env_idx=i):
                ik_failed[i] = True
            frame = build_frame(obs, dataset_action, schema=schema, prompt=prompts[i], env_idx=i)
            pending_frames[i].append(frame)
            pending_steps[i] += 1

        term = cast("torch.Tensor", terminated)
        trunc = cast("torch.Tensor", truncated)
        env_done = (term | trunc).cpu()
        max_step_done = torch.tensor([s >= max_steps for s in pending_steps], dtype=torch.bool)
        ended = env_done | max_step_done

        if not ended.any():
            obs = next_obs
            continue

        ended_indices = ended.nonzero().squeeze(-1).tolist()
        for i in ended_indices:
            attempted += 1
            attempts_since_saved += 1
            success = bool(info["success"][i].item())
            keep = success or not filter_success
            note = ""
            if ik_failed[i]:
                keep = False
                ik_discarded += 1
                print(
                    f"[ik-fallback] env={i} attempt #{attempted}: ManiSkill's CPU IK solver failed "
                    f"mid-episode, so at least one frame's action is a fabricated 'hold position' "
                    f"while the observation shows the expert moving. Episode discarded.",
                    flush=True,
                )
            if stratify and keep:
                bin_i = _stratum(i)
                if bin_counts[bin_i] >= quota:
                    keep = False
                else:
                    bin_counts[bin_i] += 1
                    note = f", bin {bin_i} {bin_counts[bin_i]}/{quota}"
            if keep and saved < num_episodes:
                for f in pending_frames[i]:
                    recorder.add_frame(f)
                recorder.save_episode(pending_provenance[i])
                saved += 1
                attempts_since_saved = 0
                tag = "success" if success else "fail"
                print(
                    f"[{saved}/{num_episodes}] saved env={i} ({tag}, {pending_steps[i]} steps, "
                    f"attempt #{attempted}{note})",
                    flush=True,
                )
            else:
                print(
                    f"[--/{num_episodes}] discarded env={i} ({pending_steps[i]} steps, attempt #{attempted})",
                    flush=True,
                )
            pending_frames[i] = []
            pending_steps[i] = 0
            pending_provenance[i] = None
            ik_failed[i] = False

        if saved == 0 and ik_discarded == attempted and attempted >= _IK_DEAD_RUN_ATTEMPTS:
            raise RuntimeError(
                f"every one of the first {attempted} episode(s) hit ManiSkill's IK fallback, so none "
                f"could be saved. `Kinematics.compute_ik` only returns None on the CPU branch, which "
                f"means this run is on a non-cuda sim device where the 100-iteration solver cannot "
                f"reach this task's poses. Collect on GPU (the LM path always returns a solution) "
                f"rather than shipping frames whose actions say 'hold position' while the expert moves."
            )

        _check_collection_stall(
            attempts_since_saved=attempts_since_saved,
            num_envs=n,
            saved=saved,
            num_episodes=num_episodes,
            bins=[
                _SpeedBinState(
                    index=b,
                    low=float(edges[b]),
                    high=float(edges[b + 1]),
                    count=bin_counts[b],
                    quota=quota,
                    **({"label": "accel", "unit": "m/s^2"} if stratify_accel else {}),
                )
                for b in range(n_bins)
            ]
            if stratify
            else None,
        )

        if saved >= num_episodes:
            break

        if stratify:
            want = int(np.argmin(bin_counts))
            if stratify_accel:
                task.randomization.accel_range = (float(edges[want]), float(edges[want + 1]))
            else:
                task.randomization.speed_range = (float(edges[want]), float(edges[want + 1]))

        if needs_periodic_reconfigure and saved - last_reconfigure_saved >= reconfigure_every:
            obs, _reset_info = env.reset(options={"reconfigure": True})
            task.reset_scripted()
            prompts = _episode_prompts()
            pending_frames = [[] for _ in range(n)]
            pending_steps = [0] * n
            ik_failed = [False] * n
            reset_index += 1
            reconfigure_index += 1
            _capture(list(range(n)))
            last_reconfigure_saved = saved
            resampled = []
            if multi_fruit:
                episode_fruits = getattr(task, "episode_fruits", [])
                resampled.append(f"fruits: {sorted(set(episode_fruits))}")
            if multi_object:
                episode_objects = getattr(task, "episode_objects", [])
                resampled.append(f"objects: {sorted(set(episode_objects))}")
            if scene_dr:
                resampled.append("scene DR")
            if multi_background:
                resampled.append(f"background: {task.background[0]}-{task.background[1]}")
            print(f"[reconfigure] resampled {', '.join(resampled)}", flush=True)
            continue

        ended_idx_t = ended.nonzero().squeeze(-1).to(task.device)
        obs, _reset_info = env.reset(options={"env_idx": ended_idx_t})
        task.reset_scripted(env_idx=ended_idx_t)
        prompts = _episode_prompts()
        reset_index += 1
        _capture(ended_indices)

    if ik_discarded:
        print(
            f"[ik-fallback] {ik_discarded} of {attempted} attempt(s) discarded: ManiSkill's CPU IK "
            f"solver failed mid-episode and the extracted actions would have said 'hold position'. "
            f"Those episodes are NOT in the dataset.",
            flush=True,
        )

    return saved, attempted


def push_to_hub(repo_id: str, folder: Path, *, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        repo_type="dataset",
        ignore_patterns=["images/"],
    )
    print(f"pushed: https://huggingface.co/datasets/{repo_id}", flush=True)
