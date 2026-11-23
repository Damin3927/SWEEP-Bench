from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_named_pick.config import RandomizationConfig
from sweep_bench.tasks.conveyor_named_pick.scripted import ScriptedNamedPickMixin
from sweep_bench.tasks.conveyor_named_pick.variants import (
    NAMED_PICK_DISTRACTOR_IDS,
    NAMED_PICK_OBJECT_IDS,
)
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

    from sweep_bench.assets.catalog import ObjectSpec


@bench_task("ConveyorNamedPick-v0")
@register_env("ConveyorNamedPick-v0", max_episode_steps=300)
class ConveyorNamedPickEnv(ScriptedNamedPickMixin, ConveyorWorkspaceEnv):
    """Pick the rider the prompt NAMES out of a short train on the belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(NAMED_PICK_OBJECT_IDS)
    DISTRACTOR_CATALOG = resolve_object_specs(NAMED_PICK_DISTRACTOR_IDS)

    _rider_pitch: float = 0.20
    _train_end_margin: float = 0.04

    _patch_radius: float = 0.02

    _grasp_margin: float = 0.022
    _lift_clearance: float = 0.12
    _target_static_speed_threshold: float = 0.10
    _carry_speed_match_threshold: float = 0.05
    _grasp_sustain_steps: int = 5

    _distractor_rise_threshold: float = 0.04
    _distractor_x_guard: float = 0.065
    _distractor_track_half_length: float = 0.85

    default_prompt: str = "pick up the red box from the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def _pick_distractor_ids(self, target_display: list[str]) -> list[str]:
        specs: tuple[ObjectSpec, ...] = self.DISTRACTOR_CATALOG
        ids = [spec.id for spec in specs]
        names = [display_name(spec) for spec in specs]
        drawn = self._batched_episode_rng.choice(np.array(ids), replace=True)
        out: list[str] = []
        for i, raw in enumerate(drawn):
            start = ids.index(str(raw))
            for step in range(len(ids)):
                k = (start + step) % len(ids)
                if names[k] != target_display[i]:
                    out.append(ids[k])
                    break
            else:
                raise ValueError(
                    f"every distractor candidate renders as {target_display[i]!r}; "
                    f"NAMED_PICK_DISTRACTOR_IDS needs >= 2 distinct display names"
                )
        return out

    def _build_riders(self, names: list[str], specs: dict[str, ObjectSpec], role: str) -> tuple[Actor, list[float]]:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        objs: list[Actor] = []
        patch_radii: list[float] = []
        for i, name in enumerate(names):
            spec = specs[name]
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + spec.bbox[2] + 0.001])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"{role}-{name}-{i}")
            self.remove_from_state_dict_registry(obj)
            obj.angular_damping = 5.0
            objs.append(obj)
            patch_radii.append(spec.patch_radius if spec.patch_radius is not None else self._patch_radius)
        return Actor.merge(objs, name=role), patch_radii

    def _load_objects(self, options: dict) -> None:
        target_specs = {spec.id: spec for spec in self.OBJECT_CATALOG}
        distractor_specs = {spec.id: spec for spec in self.DISTRACTOR_CATALOG}

        drawn = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._target_names: list[str] = [str(name) for name in drawn]
        self._target_display: list[str] = [display_name(target_specs[n]) for n in self._target_names]
        self._distractor_names: list[str] = self._pick_distractor_ids(self._target_display)

        self.target, target_radii = self._build_riders(self._target_names, target_specs, "target")
        self.distractor, distractor_radii = self._build_riders(self._distractor_names, distractor_specs, "distractor")
        self.add_to_state_dict_registry(self.target)
        self.add_to_state_dict_registry(self.distractor)
        self.belt.register_rider(
            self.target,
            patch_radius=torch.as_tensor(target_radii, dtype=torch.float32, device=self.device),
        )
        self.belt.register_rider(
            self.distractor,
            patch_radius=torch.as_tensor(distractor_radii, dtype=torch.float32, device=self.device),
        )

        def _tensor(values: list[float]) -> torch.Tensor:
            return torch.as_tensor(values, dtype=torch.float32, device=self.device)

        t_bbox = [target_specs[n].bbox for n in self._target_names]
        d_bbox = [distractor_specs[n].bbox for n in self._distractor_names]
        self._target_half_z = _tensor([b[2] for b in t_bbox])
        self._distractor_half_z = _tensor([b[2] for b in d_bbox])
        self._train_half_footprint = _tensor([max(t[0], t[1], d[0], d[1]) for t, d in zip(t_bbox, d_bbox, strict=True)])
        self._target_grasp_threshold = _tensor([math.sqrt(sum(h * h for h in b)) + self._grasp_margin for b in t_bbox])

        self._target_leads = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _task_actor_poses(self) -> list[torch.Tensor]:
        return self._lead_ordered(super()._task_actor_poses(), self._target_leads)

    @property
    def episode_objects(self) -> list[str]:
        return list(self._target_names)

    @property
    def episode_distractors(self) -> list[str]:
        return list(self._distractor_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"pick up the {name} from the conveyor belt" for name in self._target_display]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        rcfg = self.randomization
        oz = self.workspace_offset[2]
        deck_z = oz + self._conveyor_deck_height

        if rcfg.rider_spacing:
            pitch_np = self._episode_rng.uniform(*rcfg.spacing_range, size=b).astype(np.float32)
            pitch = torch.as_tensor(pitch_np, device=device)
        else:
            pitch = torch.full((b,), self._rider_pitch, device=device)

        if rcfg.target_slot:
            target_leads = torch.as_tensor(
                self._episode_rng.choice([False, True], size=b),
                device=device,
            )
        else:
            target_leads = torch.zeros((b,), dtype=torch.bool, device=device)
        self._target_leads[env_idx] = target_leads
        half_pitch = 0.5 * pitch
        target_offset = torch.where(target_leads, half_pitch, -half_pitch)

        upstream = self._conveyor_half_length - self._train_half_footprint[env_idx] - self._train_end_margin
        center_xy = self._sample_train_center_xy(env_idx, b, upstream - half_pitch)

        sign = self.belt.direction_sign[env_idx]
        target_pos = torch.zeros((b, 3), device=device)
        target_pos[:, 0] = center_xy[:, 0]
        target_pos[:, 1] = center_xy[:, 1] + target_offset * sign
        target_pos[:, 2] = deck_z + self._target_half_z[env_idx] + 0.001
        distractor_pos = torch.zeros((b, 3), device=device)
        distractor_pos[:, 0] = center_xy[:, 0]
        distractor_pos[:, 1] = center_xy[:, 1] - target_offset * sign
        distractor_pos[:, 2] = deck_z + self._distractor_half_z[env_idx] + 0.001

        target_q = self._sample_rider_yaw(b)
        distractor_q = self._sample_rider_yaw(b)
        self.target.set_pose(Pose.create_from_pq(target_pos, target_q))
        self.distractor.set_pose(Pose.create_from_pq(distractor_pos, distractor_q))

        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        zero = torch.zeros((b, 3), device=device)
        self.target.set_linear_velocity(vel)
        self.target.set_angular_velocity(zero)
        self.distractor.set_linear_velocity(vel)
        self.distractor.set_angular_velocity(zero)

        self._reset_named_pick_trackers(env_idx)

    def _sample_rider_yaw(self, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.rider_yaw:
            yaw = self._episode_rng.uniform(-np.pi, np.pi, size=b).astype(np.float32)
            yaw_t = torch.as_tensor(yaw, device=device)
            q[:, 0] = torch.cos(yaw_t * 0.5)
            q[:, 3] = torch.sin(yaw_t * 0.5)
        return q

    def _sample_train_center_xy(self, env_idx: torch.Tensor, b: int, upstream: torch.Tensor) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        device = self.device
        xy = torch.zeros((b, 2), device=device)
        xy[:, 0] = ox
        xy[:, 1] = oy - upstream * self.belt.direction_sign[env_idx]
        return xy

    def _reset_named_pick_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_grasp_sustain_counter"):
            self._grasp_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._distractor_disturbed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._prev_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._prev_tcp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._grasp_sustain_counter[env_idx] = 0
        self._distractor_disturbed[env_idx] = False
        self._prev_target_pos[env_idx] = self.target.pose.p[env_idx]
        self._prev_tcp_pos[env_idx] = cast("Panda", self.agent).tcp.pose.p[env_idx]

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        target_pos = self.target.pose.p

        target_vel = (target_pos - self._prev_target_pos) * self.control_freq
        rel_vel = ((target_pos - tcp_pos) - (self._prev_target_pos - self._prev_tcp_pos)) * self.control_freq
        self._prev_target_pos = target_pos.clone()
        self._prev_tcp_pos = tcp_pos.clone()

        is_grasped = torch.linalg.norm(tcp_pos - target_pos, dim=-1) < self._target_grasp_threshold
        is_lifted = target_pos[:, 2] > deck_z + self._lift_clearance
        is_static = torch.linalg.norm(target_vel, dim=-1) < self._target_static_speed_threshold
        is_carried = torch.linalg.norm(rel_vel, dim=-1) < self._carry_speed_match_threshold

        holding = is_grasped & is_lifted & is_static & is_carried
        self._grasp_sustain_counter = self._advance_sustain("grasp_sustain", self._grasp_sustain_counter, holding)
        held = self._grasp_sustain_counter >= self._grasp_sustain_steps

        distractor_pos = self.distractor.pose.p
        on_belt = torch.abs(distractor_pos[:, 1] - oy) < self._distractor_track_half_length
        disturbed_now = on_belt & (
            (distractor_pos[:, 2] > deck_z + self._distractor_rise_threshold)
            | (torch.abs(distractor_pos[:, 0] - ox) > self._distractor_x_guard)
        )
        self._distractor_disturbed = self._distractor_disturbed | disturbed_now

        return {
            "success": held & ~self._distractor_disturbed,
            "is_grasped": is_grasped,
            "is_lifted": is_lifted,
            "is_static": is_static,
            "is_carried": is_carried,
            "grasp_sustain": self._grasp_sustain_counter,
            "distractor_disturbed": self._distractor_disturbed,
            "target_z": target_pos[:, 2],
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        target_pos = self.target.pose.p
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        tcp_to_target = torch.linalg.norm(tcp_pos - target_pos, dim=-1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_target)

        lift_reward = torch.clamp((target_pos[:, 2] - deck_z) / self._lift_clearance, 0, 1)

        clean = (~info["distractor_disturbed"]).float()
        return reaching_reward + 2.0 * lift_reward * info["is_lifted"].float() * clean

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
