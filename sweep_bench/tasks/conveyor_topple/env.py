from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import carried_speed
from sweep_bench.tasks.conveyor_topple.assets import build_standing_block
from sweep_bench.tasks.conveyor_topple.config import RandomizationConfig
from sweep_bench.tasks.conveyor_topple.scripted import ScriptedToppleMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorTopple-v0")
@register_env("ConveyorTopple-v0", max_episode_steps=300)
class ConveyorToppleEnv(ScriptedToppleMixin, ConveyorWorkspaceEnv):
    """Topple a standing block travelling on the conveyor belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _block_half_x: float = 0.050
    _block_half_y: float = 0.012
    _block_half_z: float = 0.060
    _block_density: float = 500.0

    _block_patch_radius: float = 0.025

    _topple_flat_cos: float = math.sin(math.radians(20.0))
    _topple_lean_cos: float = math.cos(math.radians(20.0))
    _topple_settle_gap: float = 0.006
    _topple_belt_half_width: float = 0.08
    _topple_deck_drop: float = 0.005
    _topple_min_speed_frac: float = 0.5
    _topple_static_settle_speed: float = 0.02
    _topple_max_ang_speed: float = 1.0
    _topple_contact_radius: float = 0.08
    _topple_sustain_steps: int = 6

    default_prompt: str = "knock over the standing block on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _block_height(self) -> float:
        return 2.0 * self._block_half_z

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        builder = build_standing_block(
            self.scene,
            half_x=self._block_half_x,
            half_y=self._block_half_y,
            half_z=self._block_half_z,
            density=self._block_density,
        )
        builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._block_half_z])
        self.block = builder.build(name="standing_block")
        self.belt.register_rider(self.block, patch_radius=self._block_patch_radius)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        block_pos = self._sample_block_position(env_idx, b)
        upright_q = torch.zeros((b, 4), device=device)
        upright_q[:, 0] = 1.0
        self.block.set_pose(Pose.create_from_pq(block_pos, upright_q))
        block_vel = torch.zeros((b, 3), device=device)
        block_vel[:, 1] = self.belt.velocity[env_idx]
        self.block.set_linear_velocity(block_vel)
        self.block.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._sample_contact_height(env_idx, b)
        self._reset_topple_trackers(env_idx)

    def _sample_block_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._block_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.rider_lateral_offset:
            lo, hi = self.randomization.lateral_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = pos[:, 0] + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._block_half_z
        return pos

    def _sample_contact_height(self, env_idx: torch.Tensor, b: int) -> None:
        if not hasattr(self, "_block_contact_frac"):
            self._block_contact_frac = torch.full(
                (self.num_envs,), self._scripted_contact_frac, device=self.device, dtype=torch.float32
            )
        if self.randomization.contact_height:
            lo, hi = self.randomization.contact_height_range
            frac = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            self._block_contact_frac[env_idx] = torch.as_tensor(frac, device=self.device)
        else:
            self._block_contact_frac[env_idx] = self._scripted_contact_frac

    def _block_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rot = quaternion_to_matrix(self.block.pose.q)
        half = torch.as_tensor(
            [self._block_half_x, self._block_half_y, self._block_half_z],
            device=self.device,
            dtype=rot.dtype,
        )
        up_in_body = rot[:, 2, :]
        x_in_body = rot[:, 0, :]
        half_z_extent = (half * up_in_body.abs()).sum(dim=-1)
        half_x_extent = (half * x_in_body.abs()).sum(dim=-1)
        bottom_z = self.block.pose.p[:, 2] - half_z_extent
        return up_in_body[:, 2].abs(), bottom_z, half_x_extent

    def _reset_topple_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_topple_sustain_counter"):
            zeros = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            falses = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._topple_sustain_counter = zeros
            self._topple_lean_seen = falses.clone()
            self._topple_contact_caused = falses.clone()
            self._topple_tcp_was_near = falses.clone()
        self._topple_sustain_counter[env_idx] = 0
        self._topple_lean_seen[env_idx] = False
        self._topple_contact_caused[env_idx] = False
        self._topple_tcp_was_near[env_idx] = False

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        up_dot, bottom_z, half_fp_x = self._block_frame()
        block_pos = self.block.pose.p

        flat = up_dot < self._topple_flat_cos
        settled_down = (bottom_z - deck_z).abs() < self._topple_settle_gap

        on_belt = (
            (torch.abs(block_pos[:, 0] - ox) + half_fp_x < self._topple_belt_half_width)
            & (block_pos[:, 2] > deck_z - self._topple_deck_drop)
            & (torch.abs(block_pos[:, 1] - oy) < self._conveyor_half_length)
        )

        belt_speed = torch.abs(self.belt.velocity)
        along_v = self.block.linear_velocity[:, 1] * self.belt.direction_sign
        travelling = torch.where(
            belt_speed > 1e-6,
            along_v >= carried_speed(belt_speed, self._topple_min_speed_frac),
            along_v.abs() <= self._topple_static_settle_speed,
        )

        ang_settled = torch.linalg.norm(self.block.angular_velocity, dim=-1) < self._topple_max_ang_speed

        agent = cast("Panda", self.agent)
        tcp_dist = torch.linalg.norm(agent.tcp.pose.p - block_pos, dim=-1)
        near = tcp_dist < self._topple_contact_radius
        unaided = ~near

        leaning = up_dot < self._topple_lean_cos
        first_lean = leaning & ~self._topple_lean_seen
        self._topple_contact_caused = torch.where(first_lean, self._topple_tcp_was_near, self._topple_contact_caused)
        self._topple_lean_seen = self._topple_lean_seen | leaning
        self._topple_tcp_was_near = near

        holding = flat & settled_down & on_belt & travelling & ang_settled & unaided
        self._topple_sustain_counter = self._advance_sustain("topple_sustain", self._topple_sustain_counter, holding)
        sustained = self._topple_sustain_counter >= self._topple_sustain_steps

        success = sustained & self._topple_contact_caused

        return {
            "success": success,
            "toppled": flat,
            "up_dot": up_dot,
            "settled_down": settled_down,
            "block_on_belt": on_belt,
            "travelling": travelling,
            "arm_clear": unaided,
            "contact_caused": self._topple_contact_caused,
            "topple_sustain": self._topple_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        block_pos = self.block.pose.p

        tcp_to_block = torch.linalg.norm(tcp_pos - block_pos, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_block)

        topple_reward = (1.0 - info["up_dot"]).clamp(0.0, 1.0)

        return reach_reward + 2.0 * topple_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
