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
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import tolerance_speed
from sweep_bench.tasks.conveyor_flap_post.assets import build_flap_bin, build_flap_door
from sweep_bench.tasks.conveyor_flap_post.config import RandomizationConfig
from sweep_bench.tasks.conveyor_flap_post.scripted import ScriptedFlapPostMixin
from sweep_bench.tasks.conveyor_flap_post.variants import FLAP_POST_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

    from sweep_bench.assets.catalog import ObjectSpec


@bench_task("ConveyorFlapPost-v0")
@register_env("ConveyorFlapPost-v0", max_episode_steps=300)
class ConveyorFlapPostEnv(ScriptedFlapPostMixin, ConveyorWorkspaceEnv):
    """Post the held part through the spring flap of the bin riding the belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(FLAP_POST_OBJECT_IDS)

    _bin_half_x: float = 0.064
    _bin_half_y: float = 0.061
    _wall_half_thickness: float = 0.004
    _floor_half_thickness: float = 0.004
    _wall_top_z: float = 0.058
    _pin_y: float = 0.053
    _pin_z: float = 0.082
    _pin_radius: float = 0.004
    _pin_clearance: float = 0.0006
    _journal_inner_x: float = 0.056
    _journal_outer_x: float = 0.064
    _journal_base_z: float = 0.052
    _door_stop_inset: float = 0.007
    _door_stop_top_z: float = 0.044
    _bin_density: float = 1200.0

    _plate_half_x: float = 0.055
    _plate_half_y: float = 0.01875
    _plate_half_thickness: float = 0.003
    _plate_root_gap: float = 0.0015
    _pin_half_length: float = 0.065
    _door_density: float = 700.0
    _bearing_friction: float = 0.05

    _flap_return_torque: float = 0.008
    _flap_torque_ramp_angle: float = 0.11
    _flap_torque_damping: float = 0.0012
    _flap_torque_max: float = 0.03

    _bin_patch_radius: float = 0.045
    _part_patch_radius: float = 0.015

    _flap_below_plane: float = 0.03
    _flap_footprint_margin: float = 0.005
    _flap_closed_angle: float = math.radians(10.0)
    _released_finger_margin: float = 0.01
    _flap_tcp_above_hinge: float = 0.05
    _flap_tcp_part_clearance: float = 0.08
    _flap_comove_tol_base: float = 0.04
    _flap_comove_tol_per_speed: float = 0.10
    _flap_pin_capture: float = 0.010
    _bin_x_tol: float = 0.05
    _bin_track_tol_base: float = 0.06
    _bin_track_tol_per_speed: float = 0.20
    _bin_tilt_cos: float = math.cos(math.radians(10.0))
    _flap_sustain_steps: int = 6

    default_prompt: str = "post the part through the flap of the bin on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _interior_half_x(self) -> float:
        return self._bin_half_x - 2.0 * self._wall_half_thickness

    @property
    def _interior_half_y(self) -> float:
        return self._bin_half_y - 2.0 * self._wall_half_thickness

    @property
    def _interior_floor_z(self) -> float:
        return 2.0 * self._floor_half_thickness

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        bin_builder = build_flap_bin(
            self.scene,
            half_x=self._bin_half_x,
            half_y=self._bin_half_y,
            wall_half_thickness=self._wall_half_thickness,
            floor_half_thickness=self._floor_half_thickness,
            wall_top_z=self._wall_top_z,
            pin_y=self._pin_y,
            pin_z=self._pin_z,
            pin_radius=self._pin_radius,
            pin_clearance=self._pin_clearance,
            journal_inner_x=self._journal_inner_x,
            journal_outer_x=self._journal_outer_x,
            journal_base_z=self._journal_base_z,
            stop_inset=self._door_stop_inset,
            stop_top_z=self._door_stop_top_z,
            plate_half_x=self._plate_half_x,
            density=self._bin_density,
            bearing_friction=self._bearing_friction,
        )
        bin_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z])
        self.flap_bin = bin_builder.build(name="flap_bin")
        self.belt.register_rider(self.flap_bin, patch_radius=self._bin_patch_radius)

        self.doors: list[Actor] = []
        for i in range(2):
            door_builder = build_flap_door(
                self.scene,
                plate_half_x=self._plate_half_x,
                plate_half_y=self._plate_half_y,
                plate_half_thickness=self._plate_half_thickness,
                plate_root_gap=self._plate_root_gap,
                pin_radius=self._pin_radius,
                pin_half_length=self._pin_half_length,
                density=self._door_density,
                bearing_friction=self._bearing_friction,
            )
            sy = -1.0 if i == 0 else 1.0
            door_builder.initial_pose = sapien.Pose(p=[ox, oy + sy * self._pin_y, deck_z + self._pin_z])
            self.doors.append(door_builder.build(name=f"flap_door_{i}"))

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._part_names: list[str] = [str(name) for name in names]
        self._part_display: list[str] = [display_name(spec_by_id[name]) for name in self._part_names]
        specs = [spec_by_id[name] for name in self._part_names]
        self.part = self._build_part_group(specs, "posted_part", deck_z + 0.50)
        self.belt.register_rider(
            self.part,
            patch_radius=torch.as_tensor(
                [s.patch_radius if s.patch_radius is not None else self._part_patch_radius for s in specs],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        self._part_half_z = torch.as_tensor([s.bbox[2] for s in specs], dtype=torch.float32, device=self.device)
        self._part_half_fp = torch.as_tensor(
            [max(s.bbox[0], s.bbox[1]) for s in specs], dtype=torch.float32, device=self.device
        )
        self._part_grasp_width = torch.as_tensor(
            [s.grasp_width if s.grasp_width is not None else 2.0 * max(s.bbox[0], s.bbox[1]) for s in specs],
            dtype=torch.float32,
            device=self.device,
        )
        self._part_grasp_half = self._part_grasp_width / 2.0

    def _build_part_group(self, specs: list[ObjectSpec], group: str, park_z: float) -> Actor:
        ox, oy, _oz = self.workspace_offset
        objs: list[Actor] = []
        for i, spec in enumerate(specs):
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, park_z])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"{group}-{spec.id}-{i}")
            self.remove_from_state_dict_registry(obj)
            objs.append(obj)
        merged = Actor.merge(objs, name=group)
        self.add_to_state_dict_registry(merged)
        return merged

    @property
    def episode_objects(self) -> list[str]:
        return list(self._part_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"post the {name} through the flap of the bin on the conveyor belt" for name in self._part_display]

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        half_w = self._part_grasp_half[env_idx].cpu().numpy()
        return 2.0 * (half_w - 0.001)[:, None]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        self._reset_flap_trackers(env_idx, b)

        bin_pos = self._sample_bin_position(env_idx, b)
        bin_q = self._sample_bin_orientation(b)
        bin_pose = Pose.create_from_pq(bin_pos, bin_q)
        self.flap_bin.set_pose(bin_pose)
        bin_vel = torch.zeros((b, 3), device=device)
        bin_vel[:, 1] = self.belt.velocity[env_idx]
        self.flap_bin.set_linear_velocity(bin_vel)
        self.flap_bin.set_angular_velocity(torch.zeros((b, 3), device=device))

        rot = quaternion_to_matrix(bin_q)
        ajar = self._sample_ajar_angle(b)
        for i, door in enumerate(self.doors):
            sy = -1.0 if i == 0 else 1.0
            pin_local = torch.zeros((b, 3), device=device)
            pin_local[:, 1] = sy * self._pin_y
            pin_local[:, 2] = self._pin_z
            door_pos = bin_pos + torch.bmm(rot, pin_local.unsqueeze(-1)).squeeze(-1)
            shut_q = quaternion_multiply(bin_q, self._door_ref_q(i, b))
            hinge_q = torch.zeros((b, 4), device=device)
            hinge_q[:, 0] = torch.cos(ajar * 0.5)
            hinge_q[:, 1] = -torch.sin(ajar * 0.5)
            door.set_pose(Pose.create_from_pq(door_pos, quaternion_multiply(shut_q, hinge_q)))
            door.set_linear_velocity(bin_vel)
            door.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pos = cast("Panda", self.agent).tcp.pose.p[env_idx]
        part_q = torch.zeros((b, 4), device=device)
        part_q[:, 0] = 1.0
        self.part.set_pose(Pose.create_from_pq(tcp_pos, part_q))
        self.part.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.part.set_angular_velocity(torch.zeros((b, 3), device=device))

    def _door_ref_q(self, index: int, b: int) -> torch.Tensor:
        q = torch.zeros((b, 4), device=self.device)
        if index == 0:
            q[:, 0] = 1.0
        else:
            q[:, 3] = 1.0
        return q

    def _sample_ajar_angle(self, b: int) -> torch.Tensor:
        if not self.randomization.flap_start_ajar:
            return torch.zeros(b, device=self.device)
        lo, hi = self.randomization.ajar_range
        draw = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        return torch.as_tensor(draw, device=self.device)

    def _sample_torque_scale(self, b: int) -> torch.Tensor:
        if not self.randomization.flap_return_torque:
            return torch.ones(b, device=self.device)
        lo, hi = self.randomization.torque_scale_range
        draw = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        return torch.as_tensor(draw, device=self.device)

    def _sample_bin_orientation(self, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.bin_yaw:
            lo, hi = self.randomization.bin_yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
            q[:, 0] = torch.cos(yaw * 0.5)
            q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_bin_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._bin_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z
        return pos

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_flap_return_torque()

    def _apply_flap_return_torque(self) -> None:
        if not getattr(self, "_flap_torque_ready", False):
            return
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()

        angles = self._door_angles()
        bin_omega = self.flap_bin.angular_velocity
        torque_buf = self.scene.px.cuda_rigid_body_torque.torch() if gpu else None
        for door, theta in zip(self.doors, angles, strict=True):
            axis = quaternion_to_matrix(door.pose.q)[:, :, 0]
            omega = ((door.angular_velocity - bin_omega) * axis).sum(dim=-1)
            ramp = (theta / self._flap_torque_ramp_angle).clamp(0.0, 1.0)
            magnitude = self._flap_return_torque * self._flap_torque_scale * ramp
            tau = (magnitude - self._flap_torque_damping * omega).clamp(-self._flap_torque_max, self._flap_torque_max)
            torque = axis * tau.unsqueeze(-1)
            if torque_buf is not None:
                torque_buf[door._body_data_index, :3] = torque
            else:
                door._bodies[0].add_force_torque(np.zeros(3, dtype=np.float32), torque.squeeze(0).cpu().numpy())
        if gpu:
            self.scene.px.gpu_apply_rigid_dynamic_torque()

    def _door_angles(self) -> list[torch.Tensor]:
        bin_inv = quaternion_invert(self.flap_bin.pose.q)
        angles: list[torch.Tensor] = []
        for i, door in enumerate(self.doors):
            ref_inv = quaternion_invert(self._door_ref_q(i, self.num_envs))
            rel = quaternion_to_matrix(quaternion_multiply(ref_inv, quaternion_multiply(bin_inv, door.pose.q)))
            phi = torch.atan2(rel[:, 2, 1], rel[:, 1, 1])
            angles.append(-phi)
        return angles

    def _door_pin_offsets(self) -> list[torch.Tensor]:
        rot_bin = quaternion_to_matrix(self.flap_bin.pose.q)
        offsets: list[torch.Tensor] = []
        for i, door in enumerate(self.doors):
            sy = -1.0 if i == 0 else 1.0
            anchor = torch.tensor([0.0, sy * self._pin_y, self._pin_z], device=self.device)
            local = torch.bmm(rot_bin.transpose(1, 2), (door.pose.p - self.flap_bin.pose.p).unsqueeze(-1)).squeeze(-1)
            offsets.append(torch.linalg.norm(local - anchor, dim=-1))
        return offsets

    def _in_bin_frame(self, world: torch.Tensor) -> torch.Tensor:
        return (self.flap_bin.pose.inv() * Pose.create_from_pq(world)).p

    def _reset_flap_trackers(self, env_idx: torch.Tensor, b: int) -> None:
        if not hasattr(self, "_flap_sustain_counter"):
            self._flap_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._flap_torque_scale = torch.ones(self.num_envs, device=self.device)
        self._flap_sustain_counter[env_idx] = 0
        self._flap_torque_scale[env_idx] = self._sample_torque_scale(b)
        self._flap_torque_ready = True

    def evaluate(self):
        agent = cast("Panda", self.agent)
        bin_pos = self.flap_bin.pose.p
        part_pos = self.part.pose.p
        belt_v = self.belt.velocity
        tol_speed = tolerance_speed(belt_v)

        part_local = self._in_bin_frame(part_pos)
        deep_enough = (self._pin_z - part_local[:, 2]) >= self._flap_below_plane
        inside_xy = (
            torch.abs(part_local[:, 0]) <= self._interior_half_x - self._part_half_fp - self._flap_footprint_margin
        ) & (torch.abs(part_local[:, 1]) <= self._interior_half_y - self._part_half_fp - self._flap_footprint_margin)
        part_inside = deep_enough & inside_xy

        angles = self._door_angles()
        doors_shut = torch.ones_like(part_inside)
        for theta in angles:
            doors_shut = doors_shut & (torch.abs(theta) <= self._flap_closed_angle)

        finger_width = self.gripper_aperture()
        tcp_pos = agent.tcp.pose.p
        tcp_part_dist = torch.linalg.norm(tcp_pos - part_pos, dim=-1)
        released = (
            (finger_width >= self._part_grasp_width + self._released_finger_margin)
            & (tcp_part_dist > self._flap_tcp_part_clearance)
            & (tcp_pos[:, 2] - (bin_pos[:, 2] + self._pin_z) >= self._flap_tcp_above_hinge)
        )

        comove_tol = self._flap_comove_tol_base + self._flap_comove_tol_per_speed * tol_speed
        comoving = torch.linalg.norm(self.part.linear_velocity - self.flap_bin.linear_velocity, dim=-1) < comove_tol

        pins_captured = torch.ones_like(part_inside)
        for offset in self._door_pin_offsets():
            pins_captured = pins_captured & (offset < self._flap_pin_capture)

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        track_tol = self._bin_track_tol_base + self._bin_track_tol_per_speed * tol_speed
        bin_on_belt = (
            (torch.abs(bin_pos[:, 0] - ox) <= self._bin_x_tol)
            & (torch.abs(self.flap_bin.linear_velocity[:, 1] - belt_v) <= track_tol)
            & (quaternion_to_matrix(self.flap_bin.pose.q)[:, 2, 2] >= self._bin_tilt_cos)
            & (bin_pos[:, 2] > deck_z - 0.005)
            & (torch.abs(bin_pos[:, 1] - oy) <= self._conveyor_half_length - self._bin_half_y)
        )

        posted = part_inside & doors_shut & released & comoving & pins_captured & bin_on_belt

        self._flap_sustain_counter = self._advance_sustain("flap_sustain", self._flap_sustain_counter, posted)
        success = self._flap_sustain_counter >= self._flap_sustain_steps

        return {
            "success": success,
            "part_inside": part_inside,
            "part_below_plane": self._pin_z - part_local[:, 2],
            "doors_shut": doors_shut,
            "door_angle_0": angles[0],
            "door_angle_1": angles[1],
            "released": released,
            "comoving": comoving,
            "pins_captured": pins_captured,
            "bin_on_belt": bin_on_belt,
            "flap_sustain": self._flap_sustain_counter,
            "tcp_part_dist": tcp_part_dist,
        }

    def compute_dense_reward(self, obs, action, info):
        part_pos = self.part.pose.p
        mouth_pos = (self.flap_bin.pose * Pose.create_from_pq(self._mouth_local())).p

        xy_dist = torch.linalg.norm(part_pos[:, :2] - mouth_pos[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(8 * xy_dist)

        depth_reward = (info["part_below_plane"] / self._flap_below_plane).clamp(0.0, 1.0)

        return carry_reward + 2.0 * depth_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0

    def _mouth_local(self) -> torch.Tensor:
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 2] = self._pin_z
        return local
