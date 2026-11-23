from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import sapien.physx
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.peg_hole import build_hole_block, build_peg
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_on_belt
from sweep_bench.tasks.conveyor_peg_pull.config import RandomizationConfig
from sweep_bench.tasks.conveyor_peg_pull.scripted import ScriptedPullMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorPegPull-v0")
@register_env("ConveyorPegPull-v0", max_episode_steps=300)
class ConveyorPegPullEnv(ScriptedPullMixin, ConveyorWorkspaceEnv):
    """Pull a pre-seated peg out of a socket block riding the conveyor belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _peg_radius: float = 0.0025
    _peg_half_length: float = 0.045

    _block_half_extent: float = 0.04
    _block_half_height: float = 0.015

    _block_patch_radius: float = 0.025

    _conveyor_half_width: float = 0.10

    _peg_friction: float = 1.5

    _peg_tip_friction: float = 0.05
    _peg_tip_low_friction_length: float = 0.03

    _pull_clearance: float = 0.01
    _grasp_distance_threshold: float = 0.06
    _peg_static_speed_threshold: float = 0.3

    default_prompt: str = "pull the peg out of the block on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        clearance: float = 0.003,
        **kwargs,
    ) -> None:
        if clearance <= 0.0:
            raise ValueError(f"clearance must be positive (got {clearance})")
        self.clearance: float = clearance
        self._hole_half_width: float = self._peg_radius + clearance
        super().__init__(*args, **kwargs)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        block_builder = build_hole_block(
            self.scene,
            hole_half_width=self._hole_half_width,
            block_half_extent=self._block_half_extent,
            block_half_height=self._block_half_height,
        )
        block_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._block_half_height + 0.001])
        self.block = block_builder.build(name="socket_block")
        self.belt.register_rider(self.block, patch_radius=self._block_patch_radius)

        peg_builder = build_peg(self.scene, radius=self._peg_radius, half_length=self._peg_half_length)
        peg_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self._split_peg_collision_materials(peg_builder)
        self.peg = peg_builder.build(name="peg")

        tip = torch.zeros((self.num_envs, 3), device=self.device)
        tip[:, 0] = self._peg_half_length
        self._peg_tip_offset = Pose.create_from_pq(p=tip)

    def _split_peg_collision_materials(self, builder) -> None:
        grip_mat = sapien.physx.PhysxMaterial(self._peg_friction, self._peg_friction, 0.0)
        tip_mat = sapien.physx.PhysxMaterial(self._peg_tip_friction, self._peg_tip_friction, 0.0)
        r, hl = self._peg_radius, self._peg_half_length
        tip_half = self._peg_tip_low_friction_length / 2
        shaft_half = hl - tip_half
        builder.collision_records.clear()
        builder.add_cylinder_collision(
            pose=sapien.Pose(p=[hl - tip_half, 0.0, 0.0]), radius=r, half_length=tip_half, material=tip_mat
        )
        builder.add_cylinder_collision(
            pose=sapien.Pose(p=[-tip_half, 0.0, 0.0]), radius=r, half_length=shaft_half, material=grip_mat
        )

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        block_q = self._sample_block_orientation(env_idx, b)

        block_pos = self._sample_block_position(env_idx, b)
        self.block.set_pose(Pose.create_from_pq(block_pos, block_q))
        block_vel = torch.zeros((b, 3), device=device)
        block_vel[:, 1] = self.belt.velocity[env_idx]
        self.block.set_linear_velocity(block_vel)
        self.block.set_angular_velocity(torch.zeros((b, 3), device=device))

        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        peg_pos = block_pos.clone()
        peg_pos[:, 2] = deck_z + self._peg_half_length
        peg_q = torch.zeros((b, 4), device=device)
        peg_q[:, 0] = np.sqrt(0.5)
        peg_q[:, 2] = np.sqrt(0.5)
        self.peg.set_pose(Pose.create_from_pq(peg_pos, peg_q))
        peg_vel = torch.zeros((b, 3), device=device)
        peg_vel[:, 1] = self.belt.velocity[env_idx]
        self.peg.set_linear_velocity(peg_vel)
        self.peg.set_angular_velocity(torch.zeros((b, 3), device=device))

    def _sample_block_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.block_yaw:
            yaw = self._episode_rng.uniform(-np.pi / 4, np.pi / 4, size=b).astype(np.float32)
            yaw_t = torch.as_tensor(yaw, device=device)
            q[:, 0] = torch.cos(yaw_t * 0.5)
            q[:, 3] = torch.sin(yaw_t * 0.5)
        return q

    def _sample_block_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._block_half_extent - 0.04
        block_pos = torch.zeros((b, 3), device=device)
        block_pos[:, 0] = ox
        block_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        block_pos[:, 2] = deck_z + self._block_half_height + 0.001
        return block_pos

    def _peg_tip_world_z(self) -> torch.Tensor:
        tip_world = self.peg.pose * self._peg_tip_offset
        return tip_world.p[:, 2]

    def evaluate(self):
        peg_pos = self.peg.pose.p
        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        peg_speed = torch.linalg.norm(self.peg.linear_velocity, dim=-1)

        tip_z = self._peg_tip_world_z()
        block_top_z = self.block.pose.p[:, 2] + self._block_half_height

        extracted = tip_z > block_top_z + self._pull_clearance
        is_grasped = torch.linalg.norm(tcp_pos - peg_pos, dim=-1) < self._grasp_distance_threshold
        peg_static = peg_speed < self._peg_static_speed_threshold

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        block_on_belt = rider_on_belt(
            pose=self.block.pose,
            linear_velocity=self.block.linear_velocity,
            belt_velocity=self.belt.velocity,
            deck_z=deck_z,
            centre_x=ox,
            centre_y=oy,
            x_tol=self._conveyor_half_width + self._block_half_extent,
            y_tol=self._conveyor_half_length + self._block_half_extent,
            rest_height=self._block_half_height,
        )

        success = extracted & is_grasped & peg_static & block_on_belt

        return {
            "success": success,
            "extracted": extracted,
            "is_grasped": is_grasped,
            "peg_static": peg_static,
            "block_on_belt": block_on_belt,
            "tip_above_mouth": tip_z - block_top_z,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        peg_pos = self.peg.pose.p

        tcp_to_peg = torch.linalg.norm(tcp_pos - peg_pos, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_peg)

        block_top_z = self.block.pose.p[:, 2] + self._block_half_height
        tip_above = self._peg_tip_world_z() - block_top_z
        extract_reward = (tip_above / self._pull_clearance).clamp(0.0, 1.0)

        return reach_reward + 2.0 * extract_reward * info["is_grasped"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
