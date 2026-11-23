from __future__ import annotations

import os
from typing import TYPE_CHECKING, TypedDict, cast

import gymnasium as gym
import torch
from mani_skill.agents.controllers.pd_ee_pose import PDEEPosController

from sweep_bench.execution import make_executor
from sweep_bench.tasks.conveyor_base.randomization_scene import scene_randomization_active

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mani_skill.agents.controllers.base_controller import CombinedController

    from sweep_bench.policies import Obs, Policy
    from sweep_bench.tasks.conveyor_base.config import AnyRandomization, PlatformStyle, SceneStyle
    from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
    from sweep_bench.tasks.conveyor_base.placements import Background


class EpisodeRecord(TypedDict):
    index: int
    success: bool
    steps: int
    early_fail_step: int | None
    belt_speed: float | None
    belt_direction: str | None
    background: Background
    ik_fallback_steps: int | None
    contact_step: int | None
    contact_belt_speed: float | None
    contact_u: float | None
    contact_offset: float | None
    closest_step: int | None
    closest_belt_speed: float | None
    closest_u: float | None
    closest_dist: float | None
    closest_offset: float | None


class EvalMetrics(TypedDict):
    success_rate: float
    mean_inference_latency_s: float | None
    mean_hold_steps: float
    hold_fraction: float | None
    mean_inferences_per_episode: float
    episode_backgrounds: list[Background]
    episodes: list[EpisodeRecord]
    record_contact: bool


_MODEL_BACKED_POLICIES = (
    "openpi",
    "openpi_ee",
    "dynamicvla",
    "fastwam",
    "vlajepa",
    "openvlaoft",
    "groot",
    "cosmospolicy",
)
_MODEL_POLICY_ROBOTS = ("panda_wristcam",)


def make_env(
    task_id: str,
    robot: str,
    num_envs: int,
    *,
    prompt: str | None = None,
    control_mode: str | None = None,
    render_mode: str | None = None,
    randomization: AnyRandomization | None = None,
    shader_pack: str = "default",
    fruits: tuple[str, ...] | None = None,
    objects: tuple[str, ...] | None = None,
    scene_style: SceneStyle | None = None,
    scene_idx: int = 0,
    platform_style: PlatformStyle = "box",
    background_pool: Sequence[Background] | None = None,
    early_fail: bool = False,
    record_contact: bool = False,
) -> gym.Env[Obs, torch.Tensor]:
    extra_kwargs: dict = {} if fruits is None else {"fruits": fruits}
    if objects is not None:
        extra_kwargs["objects"] = objects
    if background_pool is not None:
        extra_kwargs["background_pool"] = tuple(background_pool)
    if early_fail:
        extra_kwargs["early_fail"] = True
    if record_contact:
        extra_kwargs["record_contact"] = True
    if scene_style is not None:
        extra_kwargs["scene_style"] = scene_style
        if scene_style != "table":
            extra_kwargs[f"{scene_style}_scene_idx"] = scene_idx
    return cast(
        "gym.Env[Obs, torch.Tensor]",
        gym.make(
            task_id,
            robot_uids=robot,
            obs_mode="rgbd",
            num_envs=num_envs,
            prompt=prompt,
            control_mode=control_mode,
            render_mode=render_mode,
            randomization=randomization,
            shader_pack=shader_pack,
            platform_style=platform_style,
            **extra_kwargs,
        ),
    )


def build_policy(
    name: str,
    env: gym.Env,
    num_envs: int,
    host: str,
    port: int,
    prompt: str,
) -> Policy:
    if name in _MODEL_BACKED_POLICIES:
        robot = getattr(env.unwrapped, "robot_uids", None)
        if robot not in _MODEL_POLICY_ROBOTS:
            raise ValueError(
                f"policy {name!r} supports panda_wristcam only: robot_uids={robot!r} is not "
                f"one of {list(_MODEL_POLICY_ROBOTS)}. Its adapters are panda-shaped end to end "
                f"AND read a wrist camera, which plain 'panda' does not carry (see "
                f"_MODEL_BACKED_POLICIES). Drive another robot with --policy random/zero, or "
                f"with the scripted expert."
            )
    if name == "random":
        from sweep_bench.policies.random import RandomPolicy

        return RandomPolicy(env.action_space)
    if name == "zero":
        from sweep_bench.policies.zero import ZeroActionPolicy

        return ZeroActionPolicy(env.action_space, num_envs=num_envs)
    if name == "openpi":
        from sweep_bench.policies.openpi import OpenPiDroidPolicy

        return OpenPiDroidPolicy(host=host, port=port, prompt=prompt)
    if name == "openpi_ee":
        from sweep_bench.policies.openpi_ee import OpenPiEePolicy

        return OpenPiEePolicy(host=host, port=port, prompt=prompt)
    if name == "dynamicvla":
        from sweep_bench.policies.dynamicvla import DynamicVLAPolicy

        return DynamicVLAPolicy(host=host, port=port, prompt=prompt)
    if name == "fastwam":
        from sweep_bench.policies.fastwam import FastWAMPolicy

        return FastWAMPolicy(host=host, port=port, prompt=prompt)
    if name == "vlajepa":
        from sweep_bench.policies.vlajepa import VLAJEPAPolicy

        return VLAJEPAPolicy(host=host, port=port, prompt=prompt)
    if name == "openvlaoft":
        from sweep_bench.policies.openvlaoft import OpenVlaOftPolicy

        return OpenVlaOftPolicy(host=host, port=port, prompt=prompt)
    if name == "groot":
        from sweep_bench.policies.groot import GrootPolicy

        return GrootPolicy(host=host, port=port, prompt=prompt)
    if name == "cosmospolicy":
        from sweep_bench.policies.cosmospolicy import CosmosPolicyDroidPolicy

        return CosmosPolicyDroidPolicy(host=host, port=port, prompt=prompt)
    raise ValueError(f"unknown policy: {name}")


def evaluate(
    policy: Policy,
    task_id: str,
    robot: str,
    num_envs: int = 16,
    num_episodes: int = 100,
    exec_mode: str = "sync",
    inference_latency: float | None = None,
    prompt: str | None = None,
    control_mode: str | None = None,
    randomization: AnyRandomization | None = None,
    fruits: tuple[str, ...] | None = None,
    objects: tuple[str, ...] | None = None,
    scene_style: SceneStyle | None = None,
    scene_idx: int = 0,
    platform_style: PlatformStyle = "box",
    background_pool: Sequence[Background] | None = None,
    on_episode_start: Callable[[Policy, gym.Env], None] | None = None,
    on_step: Callable[[gym.Env, dict, torch.Tensor], None] | None = None,
    video_dir: str | None = None,
    seed: int | None = None,
    record_contact: bool = False,
    episode_start: int = 0,
) -> EvalMetrics:
    if episode_start < 0:
        raise ValueError(f"episode_start must be >= 0, got {episode_start}")
    if episode_start and num_envs != 1:
        raise ValueError(f"episode_start needs num_envs=1 (one episode per reset), got {num_envs}")
    scene_dr = randomization is not None and scene_randomization_active(randomization)
    rebuild_each_episode = scene_dr or background_pool is not None
    if rebuild_each_episode and num_envs != 1:
        raise ValueError(
            f"scene randomization and background pools reconfigure on every episode end, "
            f"which resets all {num_envs} envs and discards the unfinished (disproportionately "
            f"failing) episodes; run this path with num_envs=1"
        )
    render_mode = "rgb_array" if video_dir is not None else None
    env = make_env(
        task_id,
        robot,
        num_envs,
        prompt=prompt,
        control_mode=control_mode,
        render_mode=render_mode,
        randomization=randomization,
        fruits=fruits,
        objects=objects,
        scene_style=scene_style,
        scene_idx=scene_idx,
        platform_style=platform_style,
        background_pool=background_pool,
        early_fail=True,
        record_contact=record_contact,
    )
    task = cast("ConveyorWorkspaceEnv", env.unwrapped)
    episode_backgrounds: list[Background] = []
    episodes: list[EpisodeRecord] = []
    executor = make_executor(exec_mode, policy, env, inference_latency=inference_latency)
    ik_driven = isinstance(_arm_controller(task), PDEEPosController)
    ik_fallbacks = torch.zeros(num_envs, dtype=torch.long)

    success_count = 0
    episode_count = 0

    rec_frames: list = []
    rec_idx = 0
    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)

    def _grab() -> None:
        if video_dir is not None:
            frame = env.render()
            assert frame is not None
            rec_frames.append(frame[0].detach().cpu().numpy().astype("uint8"))

    if background_pool is None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset(seed=seed, options={"reconfigure": True})
    for _ in range(episode_start):
        if rebuild_each_episode:
            obs, info = env.reset(options={"reconfigure": True})
        else:
            obs, info = env.reset(options={"env_idx": torch.arange(num_envs, device=task.device)})
    executor.reset()
    if on_episode_start is not None:
        on_episode_start(policy, env)
    if render_mode is not None:
        env.render()
    _grab()
    while episode_count < num_episodes:
        with torch.no_grad():
            action = executor.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if ik_driven and _ik_fell_back(task):
            ik_fallbacks += 1
        if render_mode is not None:
            env.render()
        _grab()

        done = torch.as_tensor(terminated | truncated)
        if on_step is not None:
            on_step(env, info, done)
        if done.any():
            success_count += info["success"][done].sum().item()
            episode_count += done.sum().item()
            episode_backgrounds += [task.background] * int(done.sum().item())
            for i in done.nonzero().squeeze(-1).tolist():
                ik_steps = int(ik_fallbacks[i].item()) if ik_driven else None
                episodes.append(
                    _episode_record(task, i, episode_start + len(episodes), bool(info["success"][i].item()), ik_steps)
                )
            ik_fallbacks[done.cpu()] = 0
            if video_dir is not None:
                rec_idx += 1
                ok = bool(info["success"][done][0].item())
                fr = getattr(env.unwrapped, "episode_fruits", None)
                fruit = fr[0] if fr else "na"
                tag = "success" if ok else "fail"
                write_video(os.path.join(video_dir, f"{tag}_{rec_idx}_{fruit}.mp4"), rec_frames)
                rec_frames = []
            env_idx = done.nonzero().squeeze(-1)
            if rebuild_each_episode:
                obs, info = env.reset(options={"reconfigure": True})
                executor.reset()
            else:
                obs, info = env.reset(options={"env_idx": env_idx})
                executor.reset(env_idx=env_idx)
            if on_episode_start is not None:
                on_episode_start(policy, env)
            _grab()

    stats = executor.stats()
    getattr(executor, "close", lambda: None)()
    env.close()
    infers = stats["inferences"]
    return {
        "episode_backgrounds": episode_backgrounds,
        "episodes": episodes,
        "record_contact": record_contact,
        "success_rate": success_count / episode_count,
        "mean_inference_latency_s": (stats["inference_time_s"] / infers) if infers else None,
        "mean_hold_steps": stats["hold_steps"] / episode_count,
        "hold_fraction": (stats["hold_steps"] / stats["steps"]) if stats["steps"] else None,
        "mean_inferences_per_episode": infers / episode_count,
    }


def _episode_record(
    task: ConveyorWorkspaceEnv, env_index: int, index: int, success: bool, ik_fallback_steps: int | None
) -> EpisodeRecord:
    belts = task._velocity_belts
    velocity = float(belts[0].velocity[env_index].item()) if belts else None
    fired = task.early_fail_step
    first = -1 if fired is None else int(fired[env_index].item())
    record = task.contact_record
    touched = None if record is None or int(record.contact_step[env_index].item()) < 0 else record
    approached = None if record is None or int(record.closest_step[env_index].item()) < 0 else record
    return {
        "index": index,
        "success": success,
        "steps": int(task.elapsed_steps[env_index].item()),
        "early_fail_step": None if first < 0 else first,
        "belt_speed": None if velocity is None else abs(velocity),
        "belt_direction": None if velocity is None else ("-y" if velocity < 0.0 else "+y"),
        "background": task.background,
        "ik_fallback_steps": ik_fallback_steps,
        "contact_step": None if touched is None else int(touched.contact_step[env_index].item()),
        "contact_belt_speed": None if touched is None else float(touched.contact_belt_speed[env_index].item()),
        "contact_u": None if touched is None else float(touched.contact_u[env_index].item()),
        "contact_offset": None if touched is None else float(touched.contact_offset[env_index].item()),
        "closest_step": None if approached is None else int(approached.closest_step[env_index].item()),
        "closest_belt_speed": None if approached is None else float(approached.closest_belt_speed[env_index].item()),
        "closest_u": None if approached is None else float(approached.closest_u[env_index].item()),
        "closest_dist": None if approached is None else float(approached.closest_dist[env_index].item()),
        "closest_offset": None if approached is None else float(approached.closest_offset[env_index].item()),
    }


def _arm_controller(task: ConveyorWorkspaceEnv) -> object:
    return cast("CombinedController", task.agent.controller).controllers["arm"]


def _ik_fell_back(task: ConveyorWorkspaceEnv) -> bool:
    arm = cast("PDEEPosController", _arm_controller(task))
    return arm._target_qpos is arm._start_qpos


def write_video(path: str, frames: list, fps: int = 20) -> None:
    import cv2
    import numpy as np

    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2BGR))
    writer.release()
