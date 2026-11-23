from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.procedural import build_can
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import tolerance_speed
from sweep_bench.tasks.conveyor_side_insert.assets import build_side_keyway_block
from sweep_bench.tasks.conveyor_side_insert.config import RandomizationConfig
from sweep_bench.tasks.conveyor_side_insert.scripted import ScriptedSideInsertMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_ROD_COLOR = (0.20, 0.42, 0.85, 1.0)


@bench_task("ConveyorSideInsert-v0")
@register_env("ConveyorSideInsert-v0", max_episode_steps=300)
class ConveyorSideInsertEnv(ScriptedSideInsertMixin, ConveyorWorkspaceEnv):
    """Push a held rod sideways into the robot-facing keyway of a moving block."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _rod_radius: float = 0.005
    _rod_half_length: float = 0.030
    _rod_grasp_offset: float = 0.018

    _block_half_x: float = 0.034
    _block_half_y: float = 0.045
    _block_half_z: float = 0.028
    _flare_half_z: float = 0.022
    _flare_depth: float = 0.010

    _block_density: float = 2600.0

    _block_patch_radius: float = 0.025

    _insert_depth: float = 0.018
    _gripped_distance_threshold: float = 0.040
    _comoving_rel_speed_threshold: float = 0.3
    _shove_x_tolerance: float = 0.035
    _shove_v_tolerance_floor: float = 0.06
    _shove_v_tolerance_speed_frac: float = 0.20
    _shove_angle_tolerance: float = 10.0
    _on_deck_z_tolerance: float = 0.008
    _seat_sustain_steps: int = 5

    default_prompt: str = "insert the rod into the side keyway of the block on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        keyway_half_y: float = 0.025,
        slot_half_z: float = 0.012,
        keyway_depth: float = 0.030,
        **kwargs,
    ) -> None:
        if keyway_half_y <= self._rod_radius:
            raise ValueError(f"keyway_half_y={keyway_half_y} leaves no along-belt clearance for the rod")
        if slot_half_z <= self._rod_radius:
            raise ValueError(f"slot_half_z={slot_half_z} leaves no vertical clearance for the rod")
        seat_target = self._insert_depth + self._seat_depth_margin
        if keyway_depth <= seat_target:
            raise ValueError(
                f"keyway_depth={keyway_depth} must exceed the expert's seat target {seat_target} m "
                "(otherwise the rod bottoms out on the back wall before it is seated)"
            )
        self.keyway_half_y: float = keyway_half_y
        self.slot_half_z: float = slot_half_z
        self.keyway_depth: float = keyway_depth
        super().__init__(*args, **kwargs)

    @property
    def _tip_ahead_of_tcp(self) -> float:
        return self._rod_grasp_offset + self._rod_half_length

    @property
    def _keyway_clear_z(self) -> float:
        return self.slot_half_z - self._rod_radius

    @property
    def _keyway_clear_y(self) -> float:
        return self.keyway_half_y - self._rod_radius

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        block_builder = build_side_keyway_block(
            self.scene,
            half_x=self._block_half_x,
            half_y=self._block_half_y,
            half_z=self._block_half_z,
            slot_half_z=self.slot_half_z,
            slot_half_y=self.keyway_half_y,
            keyway_depth=self.keyway_depth,
            flare_half_z=self._flare_half_z,
            flare_depth=self._flare_depth,
            density=self._block_density,
        )
        block_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._block_half_z + 0.001])
        self.block = block_builder.build(name="keyway_block")
        self.belt.register_rider(self.block, patch_radius=self._block_patch_radius)

        rod_builder = build_can(
            self.scene,
            radius=self._rod_radius,
            half_length=self._rod_half_length,
            color=_ROD_COLOR,
        )
        rod_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.rod = rod_builder.build(name="rod")

        tip = torch.zeros((self.num_envs, 3), device=self.device)
        tip[:, 2] = self._rod_half_length
        self._rod_tip_offset = Pose.create_from_pq(p=tip)

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._rod_radius - self._scripted_hold_margin, dtype=np.float32)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        block_pos = self._sample_block_position(env_idx, b)
        block_q = torch.zeros((b, 4), device=device)
        block_q[:, 0] = 1.0
        self.block.set_pose(Pose.create_from_pq(block_pos, block_q))
        block_vel = torch.zeros((b, 3), device=device)
        block_vel[:, 1] = self.belt.velocity[env_idx]
        self.block.set_linear_velocity(block_vel)
        self.block.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pos = cast("Panda", self.agent).tcp.pose.p[env_idx]
        rod_pos = tcp_pos.clone()
        rod_pos[:, 0] = tcp_pos[:, 0] + self._rod_grasp_offset
        rod_q = torch.zeros((b, 4), device=device)
        rod_q[:, 0] = math.sqrt(0.5)
        rod_q[:, 2] = math.sqrt(0.5)
        self.rod.set_pose(Pose.create_from_pq(rod_pos, rod_q))
        self.rod.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.rod.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_seat_trackers(env_idx, block_pos, block_q)

    def _sample_block_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._block_half_y - 0.04
        block_pos = torch.zeros((b, 3), device=device)
        block_pos[:, 0] = ox
        if self.randomization.block_lateral:
            lo, hi = self.randomization.block_lateral_range
            offsets = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            block_pos[:, 0] = ox + torch.as_tensor(offsets, device=device)
        block_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        block_pos[:, 2] = deck_z + self._block_half_z + 0.001
        return block_pos

    def _rod_tip_in_block_frame(self) -> torch.Tensor:
        tip_world = self.rod.pose * self._rod_tip_offset
        return (self.block.pose.inv() * tip_world).p

    def _reset_seat_trackers(self, env_idx: torch.Tensor, block_pos: torch.Tensor, block_q: torch.Tensor) -> None:
        if not hasattr(self, "_seat_sustain_counter"):
            self._seat_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._block_spawn_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._block_spawn_q = torch.zeros((self.num_envs, 4), device=self.device)
            self._block_spawn_q[:, 0] = 1.0
        self._seat_sustain_counter[env_idx] = 0
        self._block_spawn_pos[env_idx] = block_pos
        self._block_spawn_q[env_idx] = block_q

    def evaluate(self):
        rel = self._rod_tip_in_block_frame()
        depth = rel[:, 0] + self._block_half_x
        inserted = (
            (rel[:, 2].abs() < self._keyway_clear_z)
            & (rel[:, 1].abs() < self._keyway_clear_y)
            & (depth > self._insert_depth)
        )

        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        rod_pos = self.rod.pose.p
        gripped = torch.linalg.norm(tcp_pos - rod_pos, dim=-1) < self._gripped_distance_threshold
        rel_speed = torch.linalg.norm(self.rod.linear_velocity - self.block.linear_velocity, dim=-1)
        comoving = rel_speed < self._comoving_rel_speed_threshold

        block_pos = self.block.pose.p
        block_shift_x = (block_pos[:, 0] - self._block_spawn_pos[:, 0]).abs()
        tol_speed = tolerance_speed(self.belt.velocity)
        v_tolerance = self._shove_v_tolerance_floor + self._shove_v_tolerance_speed_frac * tol_speed
        v_slip = (self.block.linear_velocity[:, 1] - self.belt.velocity).abs()
        q_dot = (self.block.pose.q * self._block_spawn_q).sum(dim=-1).abs().clamp(max=1.0)
        turn_deg = torch.rad2deg(2.0 * torch.acos(q_dot))
        not_shoved = (
            (block_shift_x < self._shove_x_tolerance)
            & (v_slip < v_tolerance)
            & (turn_deg < self._shove_angle_tolerance)
        )

        oy = self.workspace_offset[1]
        on_deck = ((block_pos[:, 2] - self._block_spawn_pos[:, 2]).abs() < self._on_deck_z_tolerance) & (
            (block_pos[:, 1] - oy).abs() < self._conveyor_half_length + self._block_half_y
        )

        holding = inserted & gripped & comoving & not_shoved & on_deck
        self._seat_sustain_counter = self._advance_sustain("seat_sustain", self._seat_sustain_counter, holding)
        success = self._seat_sustain_counter >= self._seat_sustain_steps

        return {
            "success": success,
            "inserted": inserted,
            "gripped": gripped,
            "comoving": comoving,
            "not_shoved": not_shoved,
            "on_deck": on_deck,
            "insert_depth": depth,
            "tip_offset_z": rel[:, 2],
            "tip_offset_y": rel[:, 1],
            "block_shift_x": block_shift_x,
            "block_turn_deg": turn_deg,
            "block_v_slip": v_slip,
        }

    def compute_dense_reward(self, obs, action, info):
        rel = self._rod_tip_in_block_frame()

        err = torch.sqrt((rel[:, 2] / self._keyway_clear_z) ** 2 + (rel[:, 1] / self._keyway_clear_y) ** 2 + 1e-9)
        align_reward = 1 - torch.tanh(2.0 * err)

        depth = (rel[:, 0] + self._block_half_x).clamp(0.0, self._insert_depth) / self._insert_depth
        aligned = ((rel[:, 2].abs() < self._keyway_clear_z) & (rel[:, 1].abs() < self._keyway_clear_y)).float()
        return align_reward + 2.0 * depth * aligned + 2.0 * (info["inserted"] & info["not_shoved"]).float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
