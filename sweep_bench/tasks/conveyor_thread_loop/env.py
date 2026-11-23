from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.procedural import build_can
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_comoves, rider_on_belt
from sweep_bench.tasks.conveyor_thread_loop.assets import build_loop_carrier, build_ring
from sweep_bench.tasks.conveyor_thread_loop.config import RandomizationConfig
from sweep_bench.tasks.conveyor_thread_loop.scripted import ScriptedThreadLoopMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_ROD_COLOR = (0.20, 0.42, 0.85, 1.0)


@bench_task("ConveyorThreadLoop-v0")
@register_env("ConveyorThreadLoop-v0", max_episode_steps=300)
class ConveyorThreadLoopEnv(ScriptedThreadLoopMixin, ConveyorWorkspaceEnv):
    """Pass a held rod down through a moving ring and back out WITHOUT touching it."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _rod_radius: float = 0.006
    _rod_half_length: float = 0.080
    _rod_grasp_offset: float = 0.050

    _plate_half_x: float = 0.056
    _plate_half_y: float = 0.056
    _plate_thickness: float = 0.008
    _post_half_thickness: float = 0.005
    _ring_height: float = 0.100
    _ring_tube_radius: float = 0.005
    _ring_segments: int = 16
    _ring_density: float = 7800.0
    _ring_post_gap: float = 0.001

    _ring_weld_k: float = 250.0
    _ring_weld_k_ang: float = 0.12
    _ring_weld_deadbeat_frac: float = 0.5
    _ring_weld_force_max: float = 40.0
    _ring_weld_torque_max: float = 1.0

    _carrier_patch_radius: float = 0.030

    _carrier_spawn_max_dist: float = 0.40

    _pass_depth: float = 0.050
    _exit_height: float = 0.040
    _radial_margin: float = 0.002
    _ring_offset_tol: float = 0.004
    _grip_gap_tol: float = 0.002
    _held_distance_tol: float = 0.020
    _contact_force_threshold: float = 1e-3
    _traverse_sustain_steps: int = 5

    default_prompt: str = "pass the rod down through the ring and back out without touching it"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        inner_radius: float = 0.040,
        **kwargs,
    ) -> None:
        if inner_radius <= self._rod_radius + self._radial_margin:
            raise ValueError(
                f"inner_radius={inner_radius} leaves no clearance for a {self._rod_radius} m rod "
                f"beyond the {self._radial_margin} m predicate margin"
            )
        if inner_radius + 2.0 * self._ring_tube_radius + self._post_half_thickness > self._plate_half_x:
            raise ValueError(
                f"inner_radius={inner_radius} pushes the posts past the carrier plate (half-width {self._plate_half_x})"
            )
        self.inner_radius: float = inner_radius
        super().__init__(*args, **kwargs)

    @property
    def _plate_top_local_z(self) -> float:
        return self._plate_thickness / 2.0

    @property
    def _ring_plane_local_z(self) -> float:
        return self._plate_top_local_z + self._ring_height

    @property
    def _post_height(self) -> float:
        return self._ring_height - self._ring_tube_radius - self._ring_post_gap

    @property
    def _post_offset_x(self) -> float:
        return self.inner_radius + self._ring_tube_radius

    @property
    def _clear_radius(self) -> float:
        return self.inner_radius - self._rod_radius - self._radial_margin

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        carrier_builder = build_loop_carrier(
            self.scene,
            plate_half_x=self._plate_half_x,
            plate_half_y=self._plate_half_y,
            plate_thickness=self._plate_thickness,
            post_offset_x=self._post_offset_x,
            post_half_thickness=self._post_half_thickness,
            post_height=self._post_height,
        )
        carrier_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._plate_thickness / 2.0 + 0.001])
        self.carrier = carrier_builder.build(name="loop_carrier")
        self.belt.register_rider(self.carrier, patch_radius=self._carrier_patch_radius)

        ring_builder = build_ring(
            self.scene,
            inner_radius=self.inner_radius,
            tube_radius=self._ring_tube_radius,
            segments=self._ring_segments,
            density=self._ring_density,
        )
        ring_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._ring_plane_local_z])
        self.ring = ring_builder.build(name="loop_ring")

        rod_builder = build_can(
            self.scene,
            radius=self._rod_radius,
            half_length=self._rod_half_length,
            color=_ROD_COLOR,
        )
        rod_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.rod = rod_builder.build(name="thread_rod")

        axial = torch.zeros((self.num_envs, 3), device=self.device)
        axial[:, 2] = self._rod_half_length
        self._rod_tip_offset = Pose.create_from_pq(p=axial)
        self._rod_top_offset = Pose.create_from_pq(p=-axial)
        grip = torch.zeros((self.num_envs, 3), device=self.device)
        grip[:, 2] = -self._rod_grasp_offset
        self._rod_grip_offset = Pose.create_from_pq(p=grip)
        seat = torch.zeros((self.num_envs, 3), device=self.device)
        seat[:, 2] = self._ring_plane_local_z
        self._ring_seat_offset = Pose.create_from_pq(p=seat)

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._rod_radius - self._scripted_hold_margin, dtype=np.float32)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        carrier_pos = self._sample_carrier_position(env_idx, b)
        carrier_q = self._sample_carrier_quat(env_idx, b)
        self.carrier.set_pose(Pose.create_from_pq(carrier_pos, carrier_q))
        carrier_vel = torch.zeros((b, 3), device=device)
        carrier_vel[:, 1] = self.belt.velocity[env_idx]
        self.carrier.set_linear_velocity(carrier_vel)
        self.carrier.set_angular_velocity(torch.zeros((b, 3), device=device))

        ring_pos = carrier_pos.clone()
        ring_pos[:, 2] = carrier_pos[:, 2] + self._ring_plane_local_z
        self.ring.set_pose(Pose.create_from_pq(ring_pos, carrier_q))
        self.ring.set_linear_velocity(carrier_vel)
        self.ring.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pose = cast("Panda", self.agent).tcp.pose
        hang = torch.zeros((b, 3), device=device)
        hang[:, 2] = self._rod_grasp_offset
        in_hand = Pose.create_from_pq(p=hang)
        self.rod.set_pose(Pose.create_from_pq(tcp_pose.p[env_idx], tcp_pose.q[env_idx]) * in_hand)
        self.rod.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.rod.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_thread_trackers(env_idx)
        self._ring_weld_ready = True

    def _sample_carrier_quat(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        q = torch.zeros((b, 4), device=self.device)
        q[:, 0] = 1.0
        if self.randomization.carrier_yaw:
            lo, hi = self.randomization.carrier_yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)
            q[:, 0] = torch.cos(yaw / 2.0)
            q[:, 3] = torch.sin(yaw / 2.0)
        return q

    def _sample_carrier_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = min(
            self._conveyor_half_length - self._plate_half_y - 0.04,
            self._carrier_spawn_max_dist,
        )
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.carrier_lateral:
            lo, hi = self.randomization.carrier_lateral_range
            offsets = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = ox + torch.as_tensor(offsets, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._plate_thickness / 2.0 + 0.001
        return pos

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_ring_weld()

    def _apply_ring_weld(self) -> None:
        if not getattr(self, "_ring_weld_ready", False):
            return
        from mani_skill.utils.geometry.rotation_conversions import quaternion_invert, quaternion_multiply

        device = self.device
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()
        dt = 1.0 / self.scene.sim_config.sim_freq
        mass = self.ring.mass.to(device)

        target = self.carrier.pose * self._ring_seat_offset
        pos_err = target.p - self.ring.pose.p
        vel_err = self.carrier.linear_velocity - self.ring.linear_velocity
        force = self._ring_weld_k * pos_err + (self._ring_weld_deadbeat_frac * mass / dt).unsqueeze(-1) * vel_err
        force[:, 2] = force[:, 2] + 9.81 * mass
        norm = torch.linalg.norm(force, dim=-1, keepdim=True).clamp(min=1e-9)
        force = force * torch.clamp(self._ring_weld_force_max / norm, max=1.0)

        q_err = quaternion_multiply(target.q, quaternion_invert(self.ring.pose.q))
        rot_err = 2.0 * q_err[:, 1:] * torch.sign(q_err[:, 0:1])
        inertia = torch.as_tensor(
            np.array([body.inertia for body in self.ring._bodies], dtype=np.float32).mean(axis=1),
            device=device,
        )
        omega_err = self.carrier.angular_velocity - self.ring.angular_velocity
        torque = (
            self._ring_weld_k_ang * rot_err + (self._ring_weld_deadbeat_frac * inertia / dt).unsqueeze(-1) * omega_err
        )
        t_norm = torch.linalg.norm(torque, dim=-1, keepdim=True).clamp(min=1e-9)
        torque = torque * torch.clamp(self._ring_weld_torque_max / t_norm, max=1.0)

        if gpu:
            px = self.scene.px
            px.cuda_rigid_body_force.torch()[self.ring._body_data_index, :3] = force
            px.cuda_rigid_body_torque.torch()[self.ring._body_data_index, :3] = torque
            px.gpu_apply_rigid_dynamic_force()
            px.gpu_apply_rigid_dynamic_torque()
        else:
            self.ring._bodies[0].add_force_torque(force.squeeze(0).cpu().numpy(), torque.squeeze(0).cpu().numpy())

    def _after_simulation_step(self) -> None:
        super()._after_simulation_step()
        self._latch_ring_contact()

    def _latch_ring_contact(self) -> None:
        if not getattr(self, "_ring_weld_ready", False):
            return
        force = torch.linalg.norm(self.scene.get_pairwise_contact_forces(self.rod, self.ring), dim=-1)
        self._ring_touched = self._ring_touched | (force > self._contact_force_threshold)

    def _rod_plane_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inv = self.carrier.pose.inv()
        tip = (inv * (self.rod.pose * self._rod_tip_offset)).p
        top = (inv * (self.rod.pose * self._rod_top_offset)).p
        axis = (tip - top) / (2.0 * self._rod_half_length)
        tip_z_rel = tip[:, 2] - self._ring_plane_local_z
        dz = axis[:, 2]
        dz_safe = torch.where(dz >= 0.0, dz.clamp(min=1e-6), dz.clamp(max=-1e-6))
        t = -tip_z_rel / dz_safe
        pierce = tip[:, :2] + t.unsqueeze(-1) * axis[:, :2]
        pierce_r = torch.linalg.norm(pierce, dim=-1)
        on_rod = (t <= 0.0) & (t >= -2.0 * self._rod_half_length)
        return tip_z_rel, pierce_r, on_rod

    def _rod_gripped(self) -> torch.Tensor:
        gap = self.gripper_aperture()
        gap_ok = torch.abs(gap - 2.0 * self._rod_radius) < self._grip_gap_tol
        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        grip_point = (self.rod.pose * self._rod_grip_offset).p
        in_hand = torch.linalg.norm(tcp_pos - grip_point, dim=-1) < self._held_distance_tol
        return gap_ok & in_hand

    def _reset_thread_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_ring_touched"):
            zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._ring_touched = zeros.clone()
            self._passed_down = zeros.clone()
            self._exit_sampled = zeros.clone()
            self._exit_clean = zeros.clone()
            self._traverse_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ring_touched[env_idx] = False
        self._passed_down[env_idx] = False
        self._exit_sampled[env_idx] = False
        self._exit_clean[env_idx] = False
        self._traverse_counter[env_idx] = 0

    def evaluate(self):
        tip_z_rel, pierce_r, on_rod = self._rod_plane_geometry()
        centred = pierce_r < self._clear_radius

        self._passed_down = self._passed_down | (
            (tip_z_rel <= -self._pass_depth) & centred & on_rod & self._rod_gripped()
        )

        crossing_up = self._passed_down & (tip_z_rel >= 0.0) & ~self._exit_sampled
        self._exit_clean = self._exit_clean | (crossing_up & centred)
        self._exit_sampled = self._exit_sampled | crossing_up

        above = tip_z_rel >= self._exit_height

        ring_local = (self.carrier.pose.inv() * self.ring.pose).p
        ring_offset = torch.linalg.norm(ring_local - self._ring_seat_offset.p, dim=-1)
        ring_seated = ring_offset < self._ring_offset_tol

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        carrier_on_belt = rider_on_belt(
            pose=self.carrier.pose,
            linear_velocity=self.carrier.linear_velocity,
            belt_velocity=self.belt.velocity,
            deck_z=deck_z,
            centre_x=ox,
            centre_y=oy,
            x_tol=self._conveyor_half_width + self._plate_half_x,
            y_tol=self._conveyor_half_length + self._plate_half_y,
            rest_height=self._plate_thickness / 2.0,
        )
        travelling = rider_comoves(self.carrier.linear_velocity, self.belt.velocity)

        holding = ~self._ring_touched & self._exit_clean & above & ring_seated & carrier_on_belt
        self._traverse_counter = self._advance_sustain("traverse_sustain", self._traverse_counter, holding)
        success = self._traverse_counter >= self._traverse_sustain_steps

        return {
            "success": success,
            "ring_touched": self._ring_touched,
            "passed_down": self._passed_down,
            "exit_sampled": self._exit_sampled,
            "exit_clean": self._exit_clean,
            "tip_above_ring": above,
            "ring_seated": ring_seated,
            "carrier_on_belt": carrier_on_belt,
            "carrier_travelling": travelling,
            "tip_plane_offset": tip_z_rel,
            "pierce_radius": pierce_r,
            "ring_offset": ring_offset,
        }

    def compute_dense_reward(self, obs, action, info):
        tip_z_rel, pierce_r, _on_rod = self._rod_plane_geometry()

        align_reward = 1 - torch.tanh(20.0 * pierce_r)

        centred = (pierce_r < self._clear_radius).float()
        descend = ((-tip_z_rel) / self._pass_depth).clamp(0.0, 1.0)

        exit_progress = (tip_z_rel / self._exit_height).clamp(0.0, 1.0) * info["passed_down"].float()

        return align_reward + 2.0 * descend * centred + 2.0 * exit_progress - 3.0 * info["ring_touched"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return (self.compute_dense_reward(obs, action, info) + 3.0) / 8.0
