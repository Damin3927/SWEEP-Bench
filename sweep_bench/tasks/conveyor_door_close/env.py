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
from sweep_bench.tasks.conveyor_door_close.assets import (
    build_cabinet_body,
    build_cabinet_door,
)
from sweep_bench.tasks.conveyor_door_close.config import RandomizationConfig
from sweep_bench.tasks.conveyor_door_close.scripted import ScriptedDoorCloseMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@bench_task("ConveyorDoorClose-v0")
@register_env("ConveyorDoorClose-v0", max_episode_steps=300)
class ConveyorDoorCloseEnv(ScriptedDoorCloseMixin, ConveyorWorkspaceEnv):
    """Swing a cabinet's door shut on its VERTICAL hinge until the frame holds it."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _cab_half_x: float = 0.055
    _cab_half_y: float = 0.052
    _floor_half_thickness: float = 0.012
    _wall_half_thickness: float = 0.005
    _wall_top_z: float = 0.052
    _roof_half_thickness: float = 0.003
    _cab_density: float = 2600.0

    _pin_x: float = 0.042
    _pin_radius: float = 0.004
    _pin_clearance: float = 0.0006
    _journal_wall_half: float = 0.004
    _journal_gap: float = 0.0005
    _stub_length: float = 0.008
    _cap_half_thickness: float = 0.003
    _strike_depth: float = 0.006
    _strike_half_z: float = 0.008
    _strike_hinge_gap: float = 0.030

    _plate_half_length: float = 0.0475
    _plate_half_thickness: float = 0.003
    _plate_z_lo: float = 0.020
    _plate_z_hi: float = 0.058
    _rail_r_lo: float = 0.030
    _rail_z_hi: float = 0.102
    _door_density: float = 350.0
    _bearing_friction: float = 0.05

    _door_over_centre: float = math.radians(45.0)
    _door_open_stop: float = math.radians(80.0)
    _door_shut_stop: float = 0.135
    _door_spring_k: float = 0.045
    _door_stop_k: float = 0.30
    _door_spring_damping: float = 0.006
    _door_torque_max: float = 0.10

    _cab_patch_radius: float = 0.030

    _close_success_angle: float = math.radians(8.0)
    _close_pin_capture: float = 0.004
    _close_deck_drop: float = 0.005
    _close_lateral_band: float = 0.05
    _close_speed_tol_base: float = 0.06
    _close_speed_tol_per_speed: float = 0.20
    _close_max_tilt_cos: float = math.cos(math.radians(10.0))
    _close_tcp_clear: float = 0.08
    _close_sustain_steps: int = 8

    default_prompt: str = "swing the cabinet door shut on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _pin_y(self) -> float:
        return self._cab_half_y + self._plate_half_thickness

    @property
    def _pin_z(self) -> float:
        return 0.5 * (self._plate_z_lo + self._plate_z_hi)

    @property
    def _roof_top_z(self) -> float:
        return self._wall_top_z + 2.0 * self._roof_half_thickness

    @property
    def _pin_half_length(self) -> float:
        return 0.5 * (self._plate_z_hi - self._plate_z_lo) + self._journal_gap + self._stub_length

    @property
    def _door_open_rest(self) -> float:
        k, ks = self._door_spring_k, self._door_stop_k
        return (ks * self._door_open_stop - k * self._door_over_centre) / (ks - k)

    @property
    def _leaf_center_z(self) -> float:
        return 0.5 * (self._plate_z_lo + self._plate_z_hi) - self._pin_z

    @property
    def _leaf_half_height(self) -> float:
        return 0.5 * (self._plate_z_hi - self._plate_z_lo)

    @property
    def _rail_center_z(self) -> float:
        return 0.5 * (self._roof_top_z + self._rail_z_hi) - self._pin_z

    @property
    def _rail_half_height(self) -> float:
        return 0.5 * (self._rail_z_hi - self._roof_top_z)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        cab_builder = build_cabinet_body(
            self.scene,
            half_x=self._cab_half_x,
            half_y=self._cab_half_y,
            floor_half_thickness=self._floor_half_thickness,
            wall_half_thickness=self._wall_half_thickness,
            wall_top_z=self._wall_top_z,
            roof_half_thickness=self._roof_half_thickness,
            pin_x=self._pin_x,
            pin_y=self._pin_y,
            pin_radius=self._pin_radius,
            pin_clearance=self._pin_clearance,
            journal_wall_half=self._journal_wall_half,
            journal_gap=self._journal_gap,
            stub_length=self._stub_length,
            cap_half_thickness=self._cap_half_thickness,
            strike_depth=self._strike_depth,
            strike_half_z=self._strike_half_z,
            strike_hinge_gap=self._strike_hinge_gap,
            plate_half_thickness=self._plate_half_thickness,
            plate_z_lo=self._plate_z_lo,
            plate_z_hi=self._plate_z_hi,
            density=self._cab_density,
            bearing_friction=self._bearing_friction,
        )
        cab_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z])
        self.cabinet = cab_builder.build(name="belt_cabinet")
        self.belt.register_rider(self.cabinet, patch_radius=self._cab_patch_radius)

        door_builder = build_cabinet_door(
            self.scene,
            plate_half_length=self._plate_half_length,
            plate_half_thickness=self._plate_half_thickness,
            leaf_half_height=self._leaf_half_height,
            leaf_center_z=self._leaf_center_z,
            rail_r_lo=self._rail_r_lo,
            rail_half_height=self._rail_half_height,
            rail_center_z=self._rail_center_z,
            pin_radius=self._pin_radius,
            pin_half_length=self._pin_half_length,
            density=self._door_density,
            bearing_friction=self._bearing_friction,
        )
        door_builder.initial_pose = sapien.Pose(p=[ox + self._pin_x, oy + self._pin_y, deck_z + self._pin_z])
        self.door = door_builder.build(name="cabinet_door")
        self._door_spring_ready = False

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.door]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        cab_q = self._sample_cabinet_orientation(env_idx, b)
        cab_pos = self._sample_cabinet_position(env_idx, b)
        self.cabinet.set_pose(Pose.create_from_pq(cab_pos, cab_q))
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        self.cabinet.set_linear_velocity(vel)
        self.cabinet.set_angular_velocity(torch.zeros((b, 3), device=device))

        rot = quaternion_to_matrix(cab_q)
        pin_local = torch.zeros((b, 3), device=device)
        pin_local[:, 0] = self._pin_x
        pin_local[:, 1] = self._pin_y
        pin_local[:, 2] = self._pin_z
        door_pos = cab_pos + torch.bmm(rot, pin_local.unsqueeze(-1)).squeeze(-1)
        self.door.set_pose(Pose.create_from_pq(door_pos, self._sample_door_orientation(cab_q, b)))
        self.door.set_linear_velocity(vel)
        self.door.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._door_spring_ready = True
        self._reset_close_trackers(env_idx)

    def _sample_cabinet_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        yaw_np = self._episode_rng.uniform(*self.randomization.yaw_range, size=b).astype(np.float32)
        yaw = (
            torch.as_tensor(yaw_np, device=device) if self.randomization.cabinet_yaw else torch.zeros(b, device=device)
        )
        yaw = yaw + math.pi
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = torch.cos(yaw * 0.5)
        q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_door_orientation(self, cab_q: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        start = torch.full((b,), self._door_open_rest, device=device)
        if self.randomization.door_start_angle:
            back = self._episode_rng.uniform(*self.randomization.start_angle_range, size=b).astype(np.float32)
            start = start - torch.as_tensor(back, device=device)
        hinge_q = torch.zeros((b, 4), device=device)
        hinge_q[:, 0] = torch.cos(start * 0.5)
        hinge_q[:, 3] = -torch.sin(start * 0.5)
        return quaternion_multiply(cab_q, hinge_q)

    def _sample_cabinet_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - (self._pin_y + 2.0 * self._plate_half_length) - 0.02
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.rider_lateral_offset:
            lo, hi = self.randomization.lateral_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = pos[:, 0] + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z
        return pos

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_door_return_wrench()

    def _apply_door_return_wrench(self) -> None:
        if not getattr(self, "_door_spring_ready", False):
            return
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()

        rot_c = quaternion_to_matrix(self.cabinet.pose.q)
        hinge_axis = -rot_c[:, :, 2]

        angle = self._door_hinge_angle(rot_c)
        omega = (self.door.angular_velocity * hinge_axis).sum(dim=-1)
        over_centre = self._door_spring_k * (angle - self._door_over_centre)
        end_stop = self._door_stop_k * (
            (angle - self._door_open_stop).clamp(min=0.0) + (angle - self._door_shut_stop).clamp(max=0.0)
        )
        tau = (over_centre - end_stop - self._door_spring_damping * omega).clamp(
            -self._door_torque_max, self._door_torque_max
        )
        torque = tau.unsqueeze(-1) * hinge_axis
        force = torch.zeros_like(torque)

        if gpu:
            force_buf = self.scene.px.cuda_rigid_body_force.torch()
            torque_buf = self.scene.px.cuda_rigid_body_torque.torch()
            force_buf[self.door._body_data_index, :3] = force
            torque_buf[self.door._body_data_index, :3] = torque
            torque_buf[self.cabinet._body_data_index, :3] -= torque
            self.scene.px.gpu_apply_rigid_dynamic_force()
            self.scene.px.gpu_apply_rigid_dynamic_torque()
        else:
            zero = force.squeeze(0).cpu().numpy().astype(np.float32)
            self.door._bodies[0].add_force_torque(zero, torque.squeeze(0).cpu().numpy().astype(np.float32))
            self.cabinet._bodies[0].add_force_torque(zero, -torque.squeeze(0).cpu().numpy().astype(np.float32))

    def _door_hinge_angle(self, rot_cabinet: torch.Tensor) -> torch.Tensor:
        rel = torch.bmm(rot_cabinet.transpose(1, 2), quaternion_to_matrix(self.door.pose.q))
        return torch.atan2(rel[:, 0, 1], rel[:, 0, 0])

    def _door_in_cabinet_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        rot_c = quaternion_to_matrix(self.cabinet.pose.q)
        angle = self._door_hinge_angle(rot_c)
        pin_in_cab = torch.bmm(rot_c.transpose(1, 2), (self.door.pose.p - self.cabinet.pose.p).unsqueeze(-1)).squeeze(
            -1
        )
        anchor = torch.tensor([self._pin_x, self._pin_y, self._pin_z], device=self.device)
        centre_offset = torch.linalg.norm(pin_in_cab - anchor, dim=-1)
        rot_d = quaternion_to_matrix(self.door.pose.q)
        pin_axis_cab = torch.bmm(rot_c.transpose(1, 2), rot_d[:, :, 2].unsqueeze(-1)).squeeze(-1)
        misalign_sin = torch.linalg.norm(pin_axis_cab[:, :2], dim=-1).clamp(max=1.0)
        return angle, centre_offset + self._pin_half_length * misalign_sin

    def _door_angle(self) -> torch.Tensor:
        return self._door_hinge_angle(quaternion_to_matrix(self.cabinet.pose.q))

    def _door_centroid(self) -> torch.Tensor:
        rot_d = quaternion_to_matrix(self.door.pose.q)
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 0] = -self._plate_half_length
        local[:, 2] = self._leaf_center_z
        return self.door.pose.p + torch.bmm(rot_d, local.unsqueeze(-1)).squeeze(-1)

    def _reset_close_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_close_sustain_counter"):
            self._close_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._close_sustain_counter[env_idx] = 0

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        angle, pin_offset = self._door_in_cabinet_frame()
        cab_pos = self.cabinet.pose.p

        door_shut = torch.abs(angle) <= self._close_success_angle

        pin_captured = pin_offset < self._close_pin_capture

        on_belt = (
            (torch.abs(cab_pos[:, 2] - deck_z) <= self._close_deck_drop)
            & (torch.abs(cab_pos[:, 0] - ox) <= self._close_lateral_band)
            & (torch.abs(cab_pos[:, 1] - oy) < self._conveyor_half_length + self._cab_half_y)
        )

        tol_speed = tolerance_speed(self.belt.velocity)
        running_tol = self._close_speed_tol_base + self._close_speed_tol_per_speed * tol_speed
        speed_tol = torch.where(
            torch.abs(self.belt.velocity) > 1e-6,
            torch.minimum(running_tol, 0.5 * tol_speed),
            running_tol,
        )
        tracking = torch.abs(self.cabinet.linear_velocity[:, 1] - self.belt.velocity) <= speed_tol
        upright = quaternion_to_matrix(self.cabinet.pose.q)[:, 2, 2] > self._close_max_tilt_cos
        not_shoved = tracking & upright

        agent = cast("Panda", self.agent)
        tcp_dist = torch.linalg.norm(agent.tcp.pose.p - self._door_centroid(), dim=-1)
        tcp_clear = tcp_dist >= self._close_tcp_clear

        holding = door_shut & pin_captured & on_belt & not_shoved & tcp_clear
        self._close_sustain_counter = self._advance_sustain("close_sustain", self._close_sustain_counter, holding)
        success = self._close_sustain_counter >= self._close_sustain_steps

        return {
            "success": success,
            "door_angle": angle,
            "door_shut": door_shut,
            "pin_offset": pin_offset,
            "pin_captured": pin_captured,
            "cabinet_on_belt": on_belt,
            "cabinet_not_shoved": not_shoved,
            "cabinet_lateral": cab_pos[:, 0] - ox,
            "arm_clear": tcp_clear,
            "close_sustain": self._close_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p

        tcp_to_door = torch.linalg.norm(tcp_pos - self._door_centroid(), dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_door)

        span = max(self._door_open_rest, 1e-6)
        close_reward = (1.0 - torch.abs(info["door_angle"]) / span).clamp(0.0, 1.0)

        past_crossover = (info["door_angle"] < self._door_over_centre).float()

        return reach_reward + 2.0 * close_reward + 2.0 * past_crossover

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
