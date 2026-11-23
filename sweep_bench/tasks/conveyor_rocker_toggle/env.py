from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_matrix,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_comoves, rider_on_belt
from sweep_bench.tasks.conveyor_rocker_toggle.assets import (
    build_rocker,
    build_rocker_housing,
)
from sweep_bench.tasks.conveyor_rocker_toggle.config import RandomizationConfig
from sweep_bench.tasks.conveyor_rocker_toggle.scripted import ScriptedRockerToggleMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorRockerToggle-v0")
@register_env("ConveyorRockerToggle-v0", max_episode_steps=300)
class ConveyorRockerToggleEnv(ScriptedRockerToggleMixin, ConveyorWorkspaceEnv):
    """Flip a bistable rocker switch riding the conveyor belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _well_half_x: float = 0.0315
    _well_half_y: float = 0.0265
    _wall_thickness: float = 0.011
    _side_rail_height: float = 0.012
    _end_stop_height: float = 0.008
    _floor_half_thickness: float = 0.004
    _fulcrum_height: float = 0.0075

    _rocker_half_length: float = 0.0275
    _rocker_half_width: float = 0.0225
    _rocker_half_thickness: float = 0.004
    _pad_offset: float = 0.020
    _pad_half_x: float = 0.007
    _pad_half_y: float = 0.014
    _pad_height: float = 0.004

    _detent_gain: float = 5.0e-4
    _detent_saturation_tilt: float = math.radians(6.0)
    _detent_damping: float = 2.0e-4
    _detent_torque_max: float = 1.5e-3

    _housing_patch_radius: float = 0.03

    _toggle_success_tilt: float = math.radians(8.0)
    _toggle_sustain_steps: int = 5
    _toggle_in_well_tol: float = 0.008
    _toggle_tcp_clearance: float = 0.05

    default_prompt: str = "flip the raised end of the rocker switch on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _housing_half_x(self) -> float:
        return self._well_half_x + self._wall_thickness

    @property
    def _housing_half_y(self) -> float:
        return self._well_half_y + self._wall_thickness

    @property
    def _housing_origin_above_deck(self) -> float:
        return 2.0 * self._floor_half_thickness

    @property
    def _rocker_rest_tilt(self) -> float:
        return math.asin(self._fulcrum_height / self._rocker_half_length)

    def _rocker_pad_top_local_z(self, pad_raised: bool) -> float:
        theta = self._rocker_rest_tilt
        lift = self._pad_offset * math.sin(theta) * (1.0 if pad_raised else -1.0)
        pad_top_local = self._rocker_half_thickness + self._pad_height
        return (
            self._fulcrum_height
            + self._rocker_half_thickness * math.cos(theta)
            + lift
            + pad_top_local * math.cos(theta)
        )

    @property
    def _rocker_max_top_local_z(self) -> float:
        theta = self._rocker_rest_tilt
        pad_top_local = self._rocker_half_thickness + self._pad_height
        corner = (self._pad_offset + self._pad_half_x) * math.sin(theta) + pad_top_local * math.cos(theta)
        return self._fulcrum_height + self._rocker_half_thickness * math.cos(theta) + corner

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        housing_builder = build_rocker_housing(
            self.scene,
            well_half_x=self._well_half_x,
            well_half_y=self._well_half_y,
            wall_thickness=self._wall_thickness,
            side_rail_height=self._side_rail_height,
            end_stop_height=self._end_stop_height,
            floor_half_thickness=self._floor_half_thickness,
            fulcrum_height=self._fulcrum_height,
        )
        housing_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._housing_origin_above_deck])
        self.housing = housing_builder.build(name="rocker_housing")
        self.belt.register_rider(self.housing, patch_radius=self._housing_patch_radius)

        rocker_builder = build_rocker(
            self.scene,
            half_length=self._rocker_half_length,
            half_width=self._rocker_half_width,
            half_thickness=self._rocker_half_thickness,
            pad_offset=self._pad_offset,
            pad_half_x=self._pad_half_x,
            pad_half_y=self._pad_half_y,
            pad_height=self._pad_height,
        )
        rocker_builder.initial_pose = sapien.Pose(
            p=[ox, oy, deck_z + self._housing_origin_above_deck + self._fulcrum_height]
        )
        self.rocker = rocker_builder.build(name="rocker_plate")

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        raised_side = self._sample_raised_side(env_idx, b)
        housing_q = self._sample_housing_orientation(env_idx, b, raised_side)

        housing_pos = self._sample_housing_position(env_idx, b)
        self.housing.set_pose(Pose.create_from_pq(housing_pos, housing_q))
        housing_vel = torch.zeros((b, 3), device=device)
        housing_vel[:, 1] = self.belt.velocity[env_idx]
        self.housing.set_linear_velocity(housing_vel)
        self.housing.set_angular_velocity(torch.zeros((b, 3), device=device))

        theta = self._rocker_rest_tilt
        tilt_q = torch.zeros((b, 4), device=device)
        tilt_q[:, 0] = math.cos(-theta * 0.5)
        tilt_q[:, 2] = math.sin(-theta * 0.5)
        rocker_q = quaternion_multiply(housing_q, tilt_q)

        local = torch.zeros((b, 3), device=device)
        local[:, 0] = self._pad_offset * math.cos(theta) - self._rocker_half_thickness * math.sin(theta)
        local[:, 2] = (
            self._fulcrum_height + self._rocker_half_thickness * math.cos(theta) + self._pad_offset * math.sin(theta)
        )
        rocker_pos = housing_pos + torch.matmul(quaternion_to_matrix(housing_q), local.unsqueeze(-1)).squeeze(-1)
        self.rocker.set_pose(Pose.create_from_pq(rocker_pos, rocker_q))
        rocker_vel = torch.zeros((b, 3), device=device)
        rocker_vel[:, 1] = self.belt.velocity[env_idx]
        self.rocker.set_linear_velocity(rocker_vel)
        self.rocker.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_toggle_trackers(env_idx, b)
        self._rocker_detent_ready = True

    def _sample_housing_orientation(self, env_idx: torch.Tensor, b: int, raised_side: torch.Tensor) -> torch.Tensor:
        device = self.device
        yaw = torch.where(raised_side > 0.0, torch.zeros_like(raised_side), torch.full_like(raised_side, math.pi))
        if self.randomization.housing_yaw:
            lo, hi = self.randomization.housing_yaw_range
            jitter = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            yaw = yaw + torch.as_tensor(jitter, device=device)
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = torch.cos(yaw * 0.5)
        q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_raised_side(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        if self.randomization.toggle_start_side:
            draw = self._episode_rng.choice(np.array([1.0, -1.0], dtype=np.float32), size=b)
            return torch.as_tensor(draw, device=device)
        return torch.where(env_idx % 2 == 0, 1.0, -1.0).to(dtype=torch.float32, device=device)

    def _sample_detent_scale(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        if not self.randomization.rocker_stiffness:
            return torch.ones(b, device=self.device)
        lo, hi = self.randomization.rocker_stiffness_range
        draw = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        return torch.as_tensor(draw, device=self.device)

    def _sample_housing_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._housing_half_y - 0.04
        housing_pos = torch.zeros((b, 3), device=device)
        housing_pos[:, 0] = ox
        housing_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        housing_pos[:, 2] = deck_z + self._housing_origin_above_deck
        return housing_pos

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_rocker_detent_torque()

    def _apply_rocker_detent_torque(self) -> None:
        if not getattr(self, "_rocker_detent_ready", False):
            return
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()
        axis = quaternion_to_matrix(self.housing.pose.q)[:, :, 1]
        tilt = self._rocker_tilt()
        omega = (self.rocker.angular_velocity * axis).sum(dim=-1)
        ramp = (torch.abs(tilt) / self._detent_saturation_tilt).clamp(max=1.0)
        magnitude = -torch.sign(tilt) * (self._detent_gain * self._detent_scale) * ramp
        tau = (magnitude - self._detent_damping * omega).clamp(-self._detent_torque_max, self._detent_torque_max)
        torque = axis * tau.unsqueeze(-1)
        if gpu:
            torque_buf = self.scene.px.cuda_rigid_body_torque.torch()
            torque_buf[self.rocker._body_data_index, :3] = torque
            self.scene.px.gpu_apply_rigid_dynamic_torque()
        else:
            self.rocker._bodies[0].add_force_torque(np.zeros(3, dtype=np.float32), torque.squeeze(0).cpu().numpy())

    def _rocker_tilt(self) -> torch.Tensor:
        q_rel = quaternion_multiply(quaternion_invert(self.housing.pose.q), self.rocker.pose.q)
        x_axis = quaternion_to_matrix(q_rel)[:, :, 0]
        return torch.atan2(x_axis[:, 2], x_axis[:, 0])

    @property
    def rocker_pivot_pos(self) -> torch.Tensor:
        long_axis = quaternion_to_matrix(self.rocker.pose.q)[:, :, 0]
        return self.rocker.pose.p - long_axis * self._pad_offset

    def _rocker_well_offset(self) -> torch.Tensor:
        r_h = quaternion_to_matrix(self.housing.pose.q)
        delta = (self.rocker_pivot_pos - self.housing.pose.p).unsqueeze(-1)
        local = torch.matmul(r_h.transpose(1, 2), delta).squeeze(-1)
        return torch.linalg.norm(local[:, :2], dim=-1)

    @property
    def press_point_pos(self) -> torch.Tensor:
        return self.rocker.pose.p

    def _reset_toggle_trackers(self, env_idx: torch.Tensor, b: int) -> None:
        if not hasattr(self, "_toggle_sustain_counter"):
            self._toggle_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._rocker_start_sign = torch.ones(self.num_envs, device=self.device)
            self._detent_scale = torch.ones(self.num_envs, device=self.device)
        self._toggle_sustain_counter[env_idx] = 0
        self._rocker_start_sign[env_idx] = torch.sign(self._rocker_tilt()[env_idx])
        self._detent_scale[env_idx] = self._sample_detent_scale(env_idx, b)

    def evaluate(self):
        tilt = self._rocker_tilt()
        flip = -self._rocker_start_sign * tilt
        toggled = flip >= self._toggle_success_tilt

        in_well = self._rocker_well_offset() < self._toggle_in_well_tol

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        housing_on_belt = rider_on_belt(
            pose=self.housing.pose,
            linear_velocity=self.housing.linear_velocity,
            belt_velocity=self.belt.velocity,
            deck_z=deck_z,
            centre_x=ox,
            centre_y=oy,
            x_tol=self._conveyor_half_width + self._housing_half_x,
            y_tol=self._conveyor_half_length + self._housing_half_y,
            rest_height=self._housing_origin_above_deck,
        )
        travelling = rider_comoves(self.housing.linear_velocity, self.belt.velocity)

        holding = toggled & in_well & housing_on_belt

        self._toggle_sustain_counter = self._advance_sustain("toggle_sustain", self._toggle_sustain_counter, holding)
        held = self._toggle_sustain_counter >= self._toggle_sustain_steps

        agent = cast("Panda", self.agent)
        tcp_gap = torch.linalg.norm(agent.tcp.pose.p - self.press_point_pos, dim=-1)
        hands_off = tcp_gap >= self._toggle_tcp_clearance

        success = held & hands_off

        return {
            "success": success,
            "toggled": toggled,
            "rocker_tilt": tilt,
            "toggle_progress": flip,
            "toggle_sustain": self._toggle_sustain_counter,
            "rocker_in_well": in_well,
            "housing_on_belt": housing_on_belt,
            "housing_travelling": travelling,
            "tcp_hands_off": hands_off,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p

        tcp_to_pad = torch.linalg.norm(tcp_pos - self.press_point_pos, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_pad)

        rest_tilt = self._rocker_rest_tilt
        toggle_reward = ((info["toggle_progress"] + rest_tilt) / (rest_tilt + self._toggle_success_tilt)).clamp(
            0.0, 1.0
        )

        return reach_reward + 2.0 * toggle_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
