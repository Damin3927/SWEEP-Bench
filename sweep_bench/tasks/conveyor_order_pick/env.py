from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.procedural import build_box
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_order_pick.config import RandomizationConfig
from sweep_bench.tasks.conveyor_order_pick.scripted import ScriptedOrderPickMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@bench_task("ConveyorOrderPick-v0")
@register_env("ConveyorOrderPick-v0", max_episode_steps=300)
class ConveyorOrderPickEnv(ScriptedOrderPickMixin, ConveyorWorkspaceEnv):
    """Pick the block the prompt names by its ARRIVAL ORDER on the belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    ORDINALS: tuple[str, ...] = ("first", "second")
    PROMPT_TEMPLATES: tuple[str, ...] = (
        "pick up the {ordinal} block that reaches you from the conveyor belt",
        "grab the {ordinal} block to arrive on the conveyor belt",
        "take the {ordinal} block coming toward you off the conveyor belt",
    )
    default_ordinal: str = "first"

    default_prompt: str = "pick up the first block that reaches you from the conveyor belt"

    _block_half: float = 0.02
    _block_color: tuple[float, float, float, float] = (0.25, 0.45, 0.85, 1.0)
    _patch_radius: float = 0.02

    _slot_trailing_upstream: float = 0.86
    _slot_pitch: float = 0.12

    _grasp_distance_threshold: float = 0.05
    _lift_clearance: float = 0.12
    _target_static_speed_threshold: float = 0.10
    _grasp_sustain_steps: int = 5

    _distractor_rise_threshold: float = 0.04
    _distractor_x_guard: float = 0.055
    _distractor_track_half_length: float = 0.85
    _distractor_tip_max_deg: float = 15.0

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def _build_block(self, name: str) -> Actor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        builder = build_box(
            self.scene,
            half_x=self._block_half,
            half_y=self._block_half,
            half_z=self._block_half,
            color=self._block_color,
        )
        builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._block_half + 0.001])
        block = builder.build(name=name)
        block.angular_damping = 5.0
        return block

    def _load_objects(self, options: dict) -> None:
        self.target = self._build_block("target_block")
        self.distractor = self._build_block("distractor_block")
        self.belt.register_rider(self.target, patch_radius=self._patch_radius)
        self.belt.register_rider(self.distractor, patch_radius=self._patch_radius)

        self._target_first = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._prompt_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _task_actor_poses(self) -> list[torch.Tensor]:
        return self._lead_ordered(super()._task_actor_poses(), self._target_first)

    @property
    def episode_ordinals(self) -> list[str]:
        return ["first" if first else "second" for first in self._target_first.tolist()]

    @property
    def episode_prompts(self) -> list[str]:
        return [
            self.PROMPT_TEMPLATES[idx].format(ordinal=ordinal)
            for idx, ordinal in zip(self._prompt_idx.tolist(), self.episode_ordinals, strict=True)
        ]

    def _resolve_default_ordinal(self) -> bool:
        if self.default_ordinal not in self.ORDINALS:
            raise ValueError(f"default_ordinal must be one of {self.ORDINALS}, got {self.default_ordinal!r}")
        return self.default_ordinal == "first"

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        rcfg = self.randomization
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        if rcfg.ordinal:
            target_first = torch.as_tensor(self._episode_rng.choice([False, True], size=b), device=device)
        else:
            target_first = torch.full((b,), self._resolve_default_ordinal(), dtype=torch.bool, device=device)
        self._target_first[env_idx] = target_first
        if rcfg.prompt_paraphrase:
            drawn = self._episode_rng.randint(0, len(self.PROMPT_TEMPLATES), size=b)
            self._prompt_idx[env_idx] = torch.as_tensor(drawn, dtype=torch.long, device=device)
        else:
            self._prompt_idx[env_idx] = 0

        if rcfg.slot_pitch:
            pitch_np = self._episode_rng.uniform(*rcfg.pitch_range, size=b).astype(np.float32)
            pitch = torch.as_tensor(pitch_np, device=device)
        else:
            pitch = torch.full((b,), self._slot_pitch, device=device)
        pair_x, leading_y, trailing_y = self._sample_slot_xy(env_idx, b, pitch)

        target_y = torch.where(target_first, leading_y, trailing_y)
        distractor_y = torch.where(target_first, trailing_y, leading_y)

        spawn_z = deck_z + self._block_half + 0.001
        target_pos = torch.zeros((b, 3), device=device)
        target_pos[:, 0] = pair_x
        target_pos[:, 1] = target_y
        target_pos[:, 2] = spawn_z
        distractor_pos = torch.zeros((b, 3), device=device)
        distractor_pos[:, 0] = pair_x
        distractor_pos[:, 1] = distractor_y
        distractor_pos[:, 2] = spawn_z

        target_q = self._sample_block_yaw(b)
        distractor_q = self._sample_block_yaw(b)
        self.target.set_pose(Pose.create_from_pq(target_pos, target_q))
        self.distractor.set_pose(Pose.create_from_pq(distractor_pos, distractor_q))

        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        zero = torch.zeros((b, 3), device=device)
        self.target.set_linear_velocity(vel)
        self.target.set_angular_velocity(zero)
        self.distractor.set_linear_velocity(vel)
        self.distractor.set_angular_velocity(zero)

        self._reset_order_pick_trackers(env_idx)
        self._distractor_spawn_xy[env_idx] = distractor_pos[:, :2]
        self._distractor_spawn_q[env_idx] = distractor_q

    def _sample_slot_xy(
        self, env_idx: torch.Tensor, b: int, pitch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ox, oy, _oz = self.workspace_offset
        sign = self.belt.direction_sign[env_idx]
        trailing = torch.full((b,), self._slot_trailing_upstream, device=self.device)
        return (
            torch.full((b,), ox, device=self.device),
            oy - (trailing - pitch) * sign,
            oy - trailing * sign,
        )

    def _sample_block_yaw(self, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.block_yaw:
            yaw = self._episode_rng.uniform(-np.pi, np.pi, size=b).astype(np.float32)
            yaw_t = torch.as_tensor(yaw, device=device)
            q[:, 0] = torch.cos(yaw_t * 0.5)
            q[:, 3] = torch.sin(yaw_t * 0.5)
        return q

    def _reset_order_pick_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_grasp_sustain_counter"):
            self._grasp_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._distractor_disturbed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._prev_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._distractor_spawn_xy = torch.zeros((self.num_envs, 2), device=self.device)
            self._distractor_spawn_q = torch.zeros((self.num_envs, 4), device=self.device)
            self._distractor_spawn_q[:, 0] = 1.0
        self._grasp_sustain_counter[env_idx] = 0
        self._distractor_disturbed[env_idx] = False
        self._prev_target_pos[env_idx] = self.target.pose.p[env_idx]

    def _distractor_orientation_disturbed(self) -> torch.Tensor:
        up_now = quaternion_to_matrix(self.distractor.pose.q)[:, :, 2]
        up_spawn = quaternion_to_matrix(self._distractor_spawn_q)[:, :, 2]
        return (up_now * up_spawn).sum(dim=-1) < math.cos(math.radians(self._distractor_tip_max_deg))

    def _distractor_disturbed_now(self, distractor_pos: torch.Tensor, deck_z: float) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        on_belt = torch.abs(distractor_pos[:, 1] - oy) < self._distractor_track_half_length
        return on_belt & (
            (distractor_pos[:, 2] > deck_z + self._distractor_rise_threshold)
            | (torch.abs(distractor_pos[:, 0] - ox) > self._distractor_x_guard)
            | self._distractor_orientation_disturbed()
        )

    def evaluate(self):
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        target_pos = self.target.pose.p

        target_vel = (target_pos - self._prev_target_pos) * self.control_freq
        self._prev_target_pos = target_pos.clone()

        is_grasped = agent.is_grasping(self.target) & (
            torch.linalg.norm(tcp_pos - target_pos, dim=-1) < self._grasp_distance_threshold
        )
        is_lifted = target_pos[:, 2] > deck_z + self._lift_clearance
        is_static = torch.linalg.norm(target_vel, dim=-1) < self._target_static_speed_threshold

        holding = is_grasped & is_lifted & is_static
        self._grasp_sustain_counter = self._advance_sustain("grasp_sustain", self._grasp_sustain_counter, holding)
        held = self._grasp_sustain_counter >= self._grasp_sustain_steps

        distractor_pos = self.distractor.pose.p
        self._distractor_disturbed = self._distractor_disturbed | self._distractor_disturbed_now(distractor_pos, deck_z)

        return {
            "success": held & ~self._distractor_disturbed,
            "is_grasped": is_grasped,
            "is_lifted": is_lifted,
            "is_static": is_static,
            "grasp_sustain": self._grasp_sustain_counter,
            "distractor_disturbed": self._distractor_disturbed,
            "target_first": self._target_first,
            "target_z": target_pos[:, 2],
        }

    def compute_dense_reward(self, obs, action, info):
        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        target_pos = self.target.pose.p
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        tcp_to_target = torch.linalg.norm(tcp_pos - target_pos, dim=-1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_target)

        lift_reward = torch.clamp((target_pos[:, 2] - deck_z) / self._lift_clearance, 0, 1)

        clean = (~info["distractor_disturbed"]).float()
        return reaching_reward + 2.0 * lift_reward * info["is_lifted"].float() * clean

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
