from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_at_rest_height, rider_is_level, tolerance_speed
from sweep_bench.tasks.conveyor_meter_insert.assets import build_banded_rod, build_deep_bore_block
from sweep_bench.tasks.conveyor_meter_insert.config import RandomizationConfig
from sweep_bench.tasks.conveyor_meter_insert.scripted import ScriptedMeterInsertMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


def _local_z_axis(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
        dim=-1,
    )


def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))


@bench_task("ConveyorMeterInsert-v0")
@register_env("ConveyorMeterInsert-v0", max_episode_steps=300)
class ConveyorMeterInsertEnv(ScriptedMeterInsertMixin, ConveyorWorkspaceEnv):
    """Insert a held rod into a moving blind bore until the painted band meets the mouth."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _rod_radius: float = 0.006
    _rod_half_length: float = 0.055
    _rod_grasp_offset: float = 0.015
    _band_height: float = 0.006

    _bore_depth: float = 0.055
    _floor_thickness: float = 0.006
    _block_half_extent: float = 0.040
    _chamfer: float = 0.004

    _block_patch_radius: float = 0.025

    _block_spawn_max_dist: float = 0.55

    _conveyor_half_width: float = 0.10

    _depth_band: float = 0.004
    _axis_tilt_tol: float = math.radians(8.0)
    _finger_gap_tol: float = 0.003
    _settled_rel_speed_threshold: float = 0.3
    _seat_sustain_steps: int = 6

    _block_x_tol: float = 0.035
    _block_y_span: float = 0.85
    _block_yaw_tol: float = math.radians(10.0)
    _block_comove_tol_frac: float = 0.25
    _block_comove_tol_floor: float = 0.055
    _block_comove_tol_ceiling: float = 0.100

    default_prompt: str = "insert the rod until the painted band meets the bore mouth"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        target_depth: float = 0.029,
        clear_radius: float = 0.006,
        randomization: RandomizationConfig | None = None,
        reconfiguration_freq: int | None = None,
        num_envs: int = 1,
        **kwargs,
    ) -> None:
        if clear_radius <= 0.0:
            raise ValueError(f"clear_radius must be positive (got {clear_radius})")
        randomization = randomization if randomization is not None else self.RandomizationConfig()
        lo, hi = randomization.target_depth_range
        if not 0.0 < lo <= target_depth <= hi:
            raise ValueError(f"target_depth {target_depth} must lie in the draw range {(lo, hi)}")
        if hi + self._depth_band >= self._bore_depth:
            raise ValueError(
                f"deepest target {hi} + band {self._depth_band} must stay above the {self._bore_depth} m bore floor"
            )
        self.target_depth: float = target_depth
        self.clear_radius: float = clear_radius
        self._bore_half_width_nominal: float = clear_radius + self._rod_radius
        if reconfiguration_freq is None:
            axes_on = randomization.target_depth or randomization.bore_clearance
            reconfiguration_freq = 1 if (num_envs == 1 and axes_on) else 0
        super().__init__(
            *args, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, randomization=randomization, **kwargs
        )

    @property
    def _block_half_height(self) -> float:
        return (self._bore_depth + self._floor_thickness) / 2.0

    def _draw_per_env(self, active: bool, value_range: tuple[float, float], default: float) -> np.ndarray:
        if not active:
            return np.full(self.num_envs, default, dtype=np.float32)
        lo, hi = value_range
        drawn = np.asarray(self._batched_episode_rng.uniform(lo, hi), dtype=np.float32)
        return drawn.reshape(self.num_envs)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        h = self._block_half_height

        depths = self._draw_per_env(
            self.randomization.target_depth, self.randomization.target_depth_range, self.target_depth
        )
        clears = self._draw_per_env(
            self.randomization.bore_clearance, self.randomization.bore_clear_radius_range, self.clear_radius
        )
        self._target_depth = torch.as_tensor(depths, dtype=torch.float32, device=self.device)
        self._bore_clear_radius = torch.as_tensor(clears, dtype=torch.float32, device=self.device)
        self._coaxial_radius = self._bore_clear_radius * math.sqrt(2.0)

        blocks: list[Actor] = []
        rods: list[Actor] = []
        for i in range(self.num_envs):
            block_builder = build_deep_bore_block(
                self.scene,
                bore_half_width=float(clears[i]) + self._rod_radius,
                bore_depth=self._bore_depth,
                floor_thickness=self._floor_thickness,
                block_half_extent=self._block_half_extent,
                chamfer=self._chamfer,
            )
            block_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + h + 0.001])
            block_builder.set_scene_idxs([i])
            block = block_builder.build(name=f"bore_block-{i}")
            self.remove_from_state_dict_registry(block)
            blocks.append(block)

            rod_builder = build_banded_rod(
                self.scene,
                radius=self._rod_radius,
                half_length=self._rod_half_length,
                band_offset=float(depths[i]),
                band_height=self._band_height,
            )
            rod_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
            rod_builder.set_scene_idxs([i])
            rod = rod_builder.build(name=f"banded_rod-{i}")
            self.remove_from_state_dict_registry(rod)
            rods.append(rod)

        self.block = Actor.merge(blocks, name="bore_block")
        self.add_to_state_dict_registry(self.block)
        self.belt.register_rider(self.block, patch_radius=self._block_patch_radius)

        self.rod = Actor.merge(rods, name="banded_rod")
        self.add_to_state_dict_registry(self.rod)

        tip = torch.zeros((self.num_envs, 3), device=self.device)
        tip[:, 2] = -self._rod_half_length
        self._rod_tip_offset = Pose.create_from_pq(p=tip)

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._rod_radius - self._scripted_hold_margin, dtype=np.float32)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        block_pos = self._sample_block_position(env_idx, b)
        yaw = torch.zeros(b, device=device)
        if self.randomization.bore_yaw:
            lo, hi = self.randomization.bore_yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
        block_q = torch.zeros((b, 4), device=device)
        block_q[:, 0] = torch.cos(yaw / 2.0)
        block_q[:, 3] = torch.sin(yaw / 2.0)
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
        rod_pos[:, 2] = tcp_pos[:, 2] - self._rod_grasp_offset
        rod_q = torch.zeros((b, 4), device=device)
        rod_q[:, 0] = 1.0
        self.rod.set_pose(Pose.create_from_pq(rod_pos, rod_q))
        self.rod.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.rod.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_seat_trackers(env_idx, yaw)

    def _sample_block_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = min(
            self._conveyor_half_length - self._block_half_extent * math.sqrt(2.0) - 0.04,
            self._block_spawn_max_dist,
        )
        block_pos = torch.zeros((b, 3), device=device)
        block_pos[:, 0] = ox
        block_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        block_pos[:, 2] = deck_z + self._block_half_height + 0.001
        return block_pos

    def _reset_seat_trackers(self, env_idx: torch.Tensor, spawn_yaw: torch.Tensor) -> None:
        if not hasattr(self, "_seat_sustain_counter"):
            self._seat_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._block_spawn_yaw = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._seat_sustain_counter[env_idx] = 0
        self._block_spawn_yaw[env_idx] = spawn_yaw.to(self._block_spawn_yaw.dtype)

    def _rod_tip_in_block_frame(self) -> torch.Tensor:
        tip_world = self.rod.pose * self._rod_tip_offset
        return (self.block.pose.inv() * tip_world).p

    def evaluate(self):
        rel = self._rod_tip_in_block_frame()
        h = self._block_half_height

        depth = h - rel[:, 2]
        depth_error = depth - self._target_depth
        at_depth = torch.abs(depth_error) <= self._depth_band

        lateral = torch.linalg.norm(rel[:, :2], dim=-1)
        coaxial = lateral < self._coaxial_radius

        cos_tilt = (_local_z_axis(self.rod.pose.q) * _local_z_axis(self.block.pose.q)).sum(dim=-1).clamp(-1.0, 1.0)
        upright = cos_tilt > math.cos(self._axis_tilt_tol)

        finger_gap = self.gripper_aperture()
        gripped = torch.abs(finger_gap - 2.0 * self._rod_radius) < self._finger_gap_tol

        rel_speed = torch.linalg.norm(self.rod.linear_velocity - self.block.linear_velocity, dim=-1)
        settled = rel_speed < self._settled_rel_speed_threshold

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        block_pos = self.block.pose.p
        yaw_error = torch.atan2(
            torch.sin(_yaw(self.block.pose.q) - self._block_spawn_yaw),
            torch.cos(_yaw(self.block.pose.q) - self._block_spawn_yaw),
        )
        comove_tol = torch.clamp(
            self._block_comove_tol_frac * tolerance_speed(self.belt.velocity),
            min=self._block_comove_tol_floor,
            max=self._block_comove_tol_ceiling,
        )
        block_on_belt = (
            rider_at_rest_height(self.block.pose, deck_z=deck_z, rest_height=self._block_half_height)
            & (torch.abs(block_pos[:, 0] - ox) <= self._block_x_tol)
            & (torch.abs(block_pos[:, 1] - oy) <= self._block_y_span)
            & (torch.abs(yaw_error) <= self._block_yaw_tol)
            & rider_is_level(self.block.pose)
            & (torch.abs(self.block.linear_velocity[:, 1] - self.belt.velocity) <= comove_tol)
        )

        holding = at_depth & coaxial & upright & gripped & settled & block_on_belt
        self._seat_sustain_counter = self._advance_sustain("meter_seat", self._seat_sustain_counter, holding)
        success = self._seat_sustain_counter >= self._seat_sustain_steps

        return {
            "success": success,
            "at_depth": at_depth,
            "coaxial": coaxial,
            "upright": upright,
            "gripped": gripped,
            "settled": settled,
            "block_on_belt": block_on_belt,
            "insert_depth": depth,
            "target_depth": self._target_depth,
            "depth_error": depth_error,
            "tip_lateral_offset": lateral,
            "bore_clear_radius": self._bore_clear_radius,
            "axis_tilt": torch.arccos(cos_tilt),
            "finger_gap": finger_gap,
            "block_yaw_error": yaw_error,
        }

    def compute_dense_reward(self, obs, action, info):
        rel = self._rod_tip_in_block_frame()
        h = self._block_half_height

        lateral = torch.linalg.norm(rel[:, :2], dim=-1)
        align_reward = 1 - torch.tanh(20.0 * lateral)

        depth_error = torch.abs(h - rel[:, 2] - self._target_depth)
        meter_reward = 1 - torch.tanh(20.0 * depth_error)
        aligned = (lateral < self._coaxial_radius).float()
        return align_reward + 2.0 * meter_reward * aligned

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
