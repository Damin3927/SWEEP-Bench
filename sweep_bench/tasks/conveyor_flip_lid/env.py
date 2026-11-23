from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_multiply,
    quaternion_to_matrix,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import tolerance_speed
from sweep_bench.tasks.conveyor_flip_lid.assets import (
    build_hinged_lid,
    build_hinged_lid_box,
    stop_corner_offset,
)
from sweep_bench.tasks.conveyor_flip_lid.config import RandomizationConfig
from sweep_bench.tasks.conveyor_flip_lid.scripted import ScriptedFlipLidMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@bench_task("ConveyorFlipLid-v0")
@register_env("ConveyorFlipLid-v0", max_episode_steps=300)
class ConveyorFlipLidEnv(ScriptedFlipLidMixin, ConveyorWorkspaceEnv):
    """Flip open the hinged lid of a shallow box travelling on the conveyor."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _box_half_x: float = 0.045
    _box_half_y: float = 0.045
    _floor_half_thickness: float = 0.004
    _wall_half_thickness: float = 0.002
    _pin_y: float = -0.030
    _pin_z: float = 0.025
    _pin_radius: float = 0.005
    _pin_clearance: float = 0.0006
    _journal_inner_x: float = 0.030
    _journal_outer_x: float = 0.040
    _post_inner_x: float = 0.018
    _post_outer_x: float = 0.028
    _post_half_y: float = 0.0035
    _stop_angle_deg: float = 105.0
    _stop_radius: float = 0.030
    _box_density: float = 2000.0

    _plate_half_x: float = 0.029
    _plate_half_y: float = 0.034
    _plate_half_thickness: float = 0.003
    _plate_root_gap: float = 0.005
    _rib_half_y: float = 0.004
    _rib_half_height: float = 0.012
    _rib_centre_y: float = 0.066
    _pin_half_length: float = 0.043
    _collar_radius: float = 0.008
    _collar_half_length: float = 0.0015
    _collar_x: float = 0.0420
    _lid_density: float = 350.0
    _bearing_friction: float = 0.05

    _box_patch_radius: float = 0.030

    _flip_success_angle: float = math.radians(75.0)
    _flip_pin_capture: float = 0.012
    _flip_max_ang_speed: float = 1.0
    _flip_deck_drop: float = 0.005
    _flip_speed_tol_base: float = 0.06
    _flip_speed_tol_per_speed: float = 0.20
    _flip_max_tilt_cos: float = math.cos(math.radians(10.0))
    _flip_tcp_clear: float = 0.06
    _flip_sustain_steps: int = 4

    default_prompt: str = "flip open the lid of the box on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _rim_z(self) -> float:
        return self._pin_z - self._plate_half_thickness

    @property
    def _rail_y_lo(self) -> float:
        return self._pin_y + self._collar_radius + 0.004

    @property
    def _contact_radius_y(self) -> float:
        return self._rib_centre_y + self._rib_half_y

    @property
    def _contact_radius_z(self) -> float:
        return self._plate_half_thickness + self._rib_half_height

    @property
    def _plate_centroid_y(self) -> float:
        return self._plate_root_gap + self._plate_half_y

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        stop_dy, stop_dz = stop_corner_offset(self._stop_angle_deg, self._stop_radius)
        box_builder = build_hinged_lid_box(
            self.scene,
            half_x=self._box_half_x,
            half_y=self._box_half_y,
            floor_half_thickness=self._floor_half_thickness,
            rim_z=self._rim_z,
            wall_half_thickness=self._wall_half_thickness,
            pin_y=self._pin_y,
            pin_z=self._pin_z,
            pin_radius=self._pin_radius,
            pin_clearance=self._pin_clearance,
            journal_inner_x=self._journal_inner_x,
            journal_outer_x=self._journal_outer_x,
            post_inner_x=self._post_inner_x,
            post_outer_x=self._post_outer_x,
            post_half_y=self._post_half_y,
            post_stop_dy=-stop_dy,
            post_top_z=self._pin_z + stop_dz,
            rail_y_lo=self._rail_y_lo,
            density=self._box_density,
            bearing_friction=self._bearing_friction,
        )
        box_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z])
        self.box = box_builder.build(name="hinged_lid_box")
        self.belt.register_rider(self.box, patch_radius=self._box_patch_radius)

        lid_builder = build_hinged_lid(
            self.scene,
            plate_half_x=self._plate_half_x,
            plate_half_y=self._plate_half_y,
            plate_half_thickness=self._plate_half_thickness,
            plate_root_gap=self._plate_root_gap,
            rib_half_y=self._rib_half_y,
            rib_half_height=self._rib_half_height,
            rib_centre_y=self._rib_centre_y,
            pin_radius=self._pin_radius,
            pin_half_length=self._pin_half_length,
            collar_radius=self._collar_radius,
            collar_half_length=self._collar_half_length,
            collar_x=self._collar_x,
            density=self._lid_density,
            bearing_friction=self._bearing_friction,
        )
        lid_builder.initial_pose = sapien.Pose(p=[ox, oy + self._pin_y, deck_z + self._pin_z])
        self.lid = lid_builder.build(name="hinged_lid")

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.lid]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        box_q = self._sample_box_orientation(env_idx, b)
        box_pos = self._sample_box_position(env_idx, b)
        self.box.set_pose(Pose.create_from_pq(box_pos, box_q))
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        self.box.set_linear_velocity(vel)
        self.box.set_angular_velocity(torch.zeros((b, 3), device=device))

        rot = quaternion_to_matrix(box_q)
        pin_local = torch.zeros((b, 3), device=device)
        pin_local[:, 1] = self._pin_y
        pin_local[:, 2] = self._pin_z
        lid_pos = box_pos + torch.bmm(rot, pin_local.unsqueeze(-1)).squeeze(-1)
        self.lid.set_pose(Pose.create_from_pq(lid_pos, self._sample_lid_orientation(box_q, b)))
        self.lid.set_linear_velocity(vel)
        self.lid.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_flip_trackers(env_idx)

    def _sample_box_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        yaw_np = self._episode_rng.uniform(*self.randomization.yaw_range, size=b).astype(np.float32)
        yaw = torch.as_tensor(yaw_np, device=device) if self.randomization.box_yaw else torch.zeros(b, device=device)
        yaw = yaw + torch.where(
            self.belt.direction_sign[env_idx] < 0.0, torch.full_like(yaw, math.pi), torch.zeros_like(yaw)
        )
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = torch.cos(yaw * 0.5)
        q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_lid_orientation(self, box_q: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        ajar = torch.zeros(b, device=device)
        if self.randomization.lid_start_ajar:
            ajar_np = self._episode_rng.uniform(*self.randomization.ajar_range, size=b).astype(np.float32)
            ajar = torch.as_tensor(ajar_np, device=device)
        hinge_q = torch.zeros((b, 4), device=device)
        hinge_q[:, 0] = torch.cos(ajar * 0.5)
        hinge_q[:, 1] = torch.sin(ajar * 0.5)
        return quaternion_multiply(box_q, hinge_q)

    def _sample_box_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._box_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.rider_lateral_offset:
            lo, hi = self.randomization.lateral_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = pos[:, 0] + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z
        return pos

    def _lid_in_box_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        rot_box = quaternion_to_matrix(self.box.pose.q)
        rot_lid = quaternion_to_matrix(self.lid.pose.q)
        rel = torch.bmm(rot_box.transpose(1, 2), rot_lid)
        angle = torch.arccos(rel[:, 2, 2].clamp(-1.0, 1.0))

        anchor = torch.tensor([0.0, self._pin_y, self._pin_z], device=self.device)
        pin_in_box = torch.bmm(rot_box.transpose(1, 2), (self.lid.pose.p - self.box.pose.p).unsqueeze(-1)).squeeze(-1)
        return angle, torch.linalg.norm(pin_in_box - anchor, dim=-1)

    def _flip_lid_angle(self) -> torch.Tensor:
        return self._lid_in_box_frame()[0]

    def _lid_centroid(self) -> torch.Tensor:
        rot_lid = quaternion_to_matrix(self.lid.pose.q)
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 1] = self._plate_centroid_y
        return self.lid.pose.p + torch.bmm(rot_lid, local.unsqueeze(-1)).squeeze(-1)

    def _reset_flip_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_flip_sustain_counter"):
            self._flip_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._flip_sustain_counter[env_idx] = 0

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        angle, pin_offset = self._lid_in_box_frame()
        box_pos = self.box.pose.p

        lid_open = angle >= self._flip_success_angle
        pin_captured = pin_offset < self._flip_pin_capture
        settled = torch.linalg.norm(self.lid.angular_velocity, dim=-1) < self._flip_max_ang_speed

        on_belt = (
            (box_pos[:, 2] > deck_z - self._flip_deck_drop)
            & (torch.abs(box_pos[:, 0] - ox) < self._conveyor_half_width + self._box_half_x)
            & (torch.abs(box_pos[:, 1] - oy) < self._conveyor_half_length + self._box_half_y)
        )

        speed_tol = self._flip_speed_tol_base + self._flip_speed_tol_per_speed * tolerance_speed(self.belt.velocity)
        tracking = torch.abs(self.box.linear_velocity[:, 1] - self.belt.velocity) <= speed_tol
        upright = quaternion_to_matrix(self.box.pose.q)[:, 2, 2] > self._flip_max_tilt_cos
        not_shoved = tracking & upright

        agent = cast("Panda", self.agent)
        tcp_dist = torch.linalg.norm(agent.tcp.pose.p - self._lid_centroid(), dim=-1)
        tcp_clear = tcp_dist >= self._flip_tcp_clear

        holding = lid_open & pin_captured & settled & on_belt & not_shoved & tcp_clear
        self._flip_sustain_counter = self._advance_sustain("flip_sustain", self._flip_sustain_counter, holding)
        success = self._flip_sustain_counter >= self._flip_sustain_steps

        return {
            "success": success,
            "lid_angle": angle,
            "lid_open": lid_open,
            "pin_offset": pin_offset,
            "pin_captured": pin_captured,
            "lid_settled": settled,
            "box_on_belt": on_belt,
            "box_not_shoved": not_shoved,
            "arm_clear": tcp_clear,
            "flip_sustain": self._flip_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p

        tcp_to_lid = torch.linalg.norm(tcp_pos - self._lid_centroid(), dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_lid)

        flip_reward = (info["lid_angle"] / self._flip_success_angle).clamp(0.0, 1.0)

        return reach_reward + 2.0 * flip_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
