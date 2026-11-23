from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import tolerance_speed
from sweep_bench.tasks.conveyor_even_load.assets import (
    PAD_COLORS,
    build_load_pad,
    build_twin_pad_carrier,
)
from sweep_bench.tasks.conveyor_even_load.config import RandomizationConfig
from sweep_bench.tasks.conveyor_even_load.scripted import ScriptedEvenLoadMixin
from sweep_bench.tasks.conveyor_even_load.variants import EVEN_LOAD_BAR_SPEC
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_G = 9.81

_SPRING_DAMPING_RATIO = 1.0

_BAR_IN_TCP_Q = (0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0)


@bench_task("ConveyorEvenLoad-v0")
@register_env("ConveyorEvenLoad-v0", max_episode_steps=300)
class ConveyorEvenLoadEnv(ScriptedEvenLoadMixin, ConveyorWorkspaceEnv):
    """Lay a held bar across two sprung pads so the load splits evenly between them."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _pad_spacing: float = 0.120
    _carrier_half_x: float = 0.040
    _carrier_half_y: float = 0.100
    _bore_half_width: float = 0.0206
    _carrier_wall_height: float = 0.042
    _carrier_floor_half_thickness: float = 0.004

    _pad_half_width: float = 0.020
    _pad_half_thickness: float = 0.022
    _pad_rest_gap: float = 0.020

    _spring_deflection_nominal: float = 0.006
    _spring_deflection_soft: float = 0.008
    _spring_force_max: float = 20.0

    _carrier_patch_radius: float = 0.040

    _depression_min: float = 0.003
    _depression_max: float = 0.016
    _even_delta_tol: float = 0.004
    _seated_z_tol: float = 0.004
    _bar_roll_sin_tol: float = math.sin(math.radians(6.0))
    _hands_off_finger_margin: float = 0.01
    _hands_off_distance: float = 0.10
    _carrier_tilt_cos: float = math.cos(math.radians(10.0))
    _carrier_slip_tol_base: float = 0.05
    _carrier_slip_tol_rate: float = 0.30
    _even_sustain_steps: int = 8

    default_prompt: str = "lay the bar evenly across both pads on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _carrier_origin_above_deck(self) -> float:
        return 2.0 * self._carrier_floor_half_thickness

    @property
    def _pad_anchor_local_z(self) -> float:
        return self._pad_rest_gap + self._pad_half_thickness

    @property
    def _pad_top_above_housing(self) -> float:
        return self._pad_anchor_local_z + self._pad_half_thickness

    @property
    def _bar_half_thickness(self) -> float:
        return EVEN_LOAD_BAR_SPEC.bbox[2]

    @property
    def _bar_half_length(self) -> float:
        return EVEN_LOAD_BAR_SPEC.bbox[1]

    @property
    def _bar_half_width(self) -> float:
        return EVEN_LOAD_BAR_SPEC.bbox[0]

    @property
    def _pad_local_y(self) -> tuple[float, float]:
        return (-0.5 * self._pad_spacing, 0.5 * self._pad_spacing)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        origin_z = deck_z + self._carrier_origin_above_deck

        carrier_builder = build_twin_pad_carrier(
            self.scene,
            bore_half_width=self._bore_half_width,
            pad_spacing=self._pad_spacing,
            outer_half_x=self._carrier_half_x,
            outer_half_y=self._carrier_half_y,
            wall_height=self._carrier_wall_height,
            floor_half_thickness=self._carrier_floor_half_thickness,
        )
        carrier_builder.initial_pose = sapien.Pose(p=[ox, oy, origin_z])
        self.carrier = carrier_builder.build(name="twin_pad_carrier")
        self.belt.register_rider(self.carrier, patch_radius=self._carrier_patch_radius)

        self.pads = []
        for i, local_y in enumerate(self._pad_local_y):
            pad_builder = build_load_pad(
                self.scene,
                half_width=self._pad_half_width,
                half_thickness=self._pad_half_thickness,
                color=PAD_COLORS[i],
            )
            pad_builder.initial_pose = sapien.Pose(p=[ox, oy + local_y, origin_z + self._pad_anchor_local_z])
            self.pads.append(pad_builder.build(name=f"load_pad_{i}"))

        bar_builder = EVEN_LOAD_BAR_SPEC.builder_fn(self.scene)
        bar_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.50])
        self.bar = bar_builder.build(name="transfer_bar")
        patch_radius = EVEN_LOAD_BAR_SPEC.patch_radius
        assert patch_radius is not None
        self.belt.register_rider(self.bar, patch_radius=patch_radius)
        self._check_bar_mass()

    def _check_bar_mass(self) -> None:
        built = float(self.bar.mass.max().item())
        declared = EVEN_LOAD_BAR_SPEC.mass
        if abs(built - declared) > 0.01 * declared:
            raise ValueError(
                f"transfer bar builds at {built} kg but EVEN_LOAD_BAR_SPEC declares {declared} kg; the pad "
                f"springs are derived from the declared mass, so the calibration would be wrong (see variants.py)"
            )

    def _rest_aperture(self, env_idx: torch.Tensor) -> float:
        grasp_width = EVEN_LOAD_BAR_SPEC.grasp_width
        assert grasp_width is not None
        return 2.0 * (0.5 * grasp_width - 0.001)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        carrier_q = self._sample_carrier_orientation(env_idx, b)
        carrier_pos = self._sample_carrier_position(env_idx, b)
        self.carrier.set_pose(Pose.create_from_pq(carrier_pos, carrier_q))
        carrier_vel = torch.zeros((b, 3), device=device)
        carrier_vel[:, 1] = self.belt.velocity[env_idx]
        self.carrier.set_linear_velocity(carrier_vel)
        self.carrier.set_angular_velocity(torch.zeros((b, 3), device=device))

        for pad, local_y in zip(self.pads, self._pad_local_y, strict=True):
            offset = torch.zeros((b, 3), device=device)
            offset[:, 1] = local_y
            pad_pos = (Pose.create_from_pq(carrier_pos, carrier_q) * Pose.create_from_pq(offset)).p
            pad_pos[:, 2] = carrier_pos[:, 2] + self._pad_anchor_local_z
            pad.set_pose(Pose.create_from_pq(pad_pos, carrier_q))
            pad.set_linear_velocity(carrier_vel)
            pad.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_springs(env_idx, b)

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pose = cast("Panda", self.agent).tcp.pose
        bar_q = quaternion_multiply(
            tcp_pose.q[env_idx],
            torch.as_tensor(_BAR_IN_TCP_Q, dtype=torch.float32, device=device).expand(b, 4),
        )
        self.bar.set_pose(Pose.create_from_pq(tcp_pose.p[env_idx], bar_q))
        self.bar.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.bar.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_even_trackers(env_idx)
        self._springs_ready = True

    def _sample_carrier_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.carrier_yaw:
            lo, hi = self.randomization.carrier_yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
            q[:, 0] = torch.cos(yaw * 0.5)
            q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_carrier_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._carrier_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.carrier_lateral_offset:
            lo, hi = self.randomization.carrier_x_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = ox + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._carrier_origin_above_deck
        return pos

    def _spring_stiffness(self, bar_mass: float, deflection: torch.Tensor) -> torch.Tensor:
        return 0.5 * bar_mass * _G / deflection

    def _reset_springs(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        if not hasattr(self, "_spring_k"):
            zeros = torch.zeros(self.num_envs, device=device)
            self._spring_k = zeros.clone()
            self._spring_preload = torch.zeros((len(self.pads), self.num_envs), device=device)
            self._spring_damping = zeros.clone()

        deflection = torch.full((b,), self._spring_deflection_nominal, device=device)
        if self.randomization.spring_bin:
            soft = self._episode_rng.uniform(0.0, 1.0, size=b) < 0.5
            deflection = torch.where(
                torch.as_tensor(soft, device=device),
                torch.full((b,), self._spring_deflection_soft, device=device),
                deflection,
            )
        self._spring_deflection_target = getattr(
            self, "_spring_deflection_target", torch.zeros(self.num_envs, device=device)
        )
        self._spring_deflection_target[env_idx] = deflection

        bar_mass = EVEN_LOAD_BAR_SPEC.mass
        k = self._spring_stiffness(bar_mass, deflection)
        self._spring_k[env_idx] = k
        for i, pad in enumerate(self.pads):
            self._spring_preload[i, env_idx] = pad.mass.to(device)[env_idx] * _G
        pad_mass = self.pads[0].mass.to(device)[env_idx]
        self._spring_damping[env_idx] = 2.0 * _SPRING_DAMPING_RATIO * torch.sqrt(k * pad_mass)

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_pad_spring_forces()

    def _apply_pad_spring_forces(self) -> None:
        if not getattr(self, "_springs_ready", False):
            return
        device = self.device
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()
        carrier_vz = self.carrier.linear_velocity[:, 2]
        force_buf = self.scene.px.cuda_rigid_body_force.torch() if gpu else None
        for i, pad in enumerate(self.pads):
            depression = self._pad_depression(pad)
            v_rel_z = pad.linear_velocity[:, 2] - carrier_vz
            fz = (self._spring_k * depression + self._spring_preload[i] - self._spring_damping * v_rel_z).clamp(
                0.0, self._spring_force_max
            )
            force = torch.zeros((self.num_envs, 3), device=device)
            force[:, 2] = fz
            if force_buf is not None:
                force_buf[pad._body_data_index, :3] = force
            else:
                pad._bodies[0].add_force_torque(force.squeeze(0).cpu().numpy(), np.zeros(3, dtype=np.float32))
        if gpu:
            self.scene.px.gpu_apply_rigid_dynamic_force()

    def _pad_depression(self, pad) -> torch.Tensor:
        rel_z = pad.pose.p[:, 2] - self.carrier.pose.p[:, 2]
        return self._pad_anchor_local_z - rel_z

    def _reset_even_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_even_sustain_counter"):
            self._even_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._even_sustain_counter[env_idx] = 0

    def _local_axis(self, pose: Pose, axis: int) -> torch.Tensor:
        unit = torch.zeros((self.num_envs, 3), device=self.device)
        unit[:, axis] = 1.0
        return (pose * Pose.create_from_pq(unit)).p - pose.p

    def evaluate(self):
        agent = cast("Panda", self.agent)
        carrier_pose = self.carrier.pose
        carrier_pos = carrier_pose.p
        bar_pose = self.bar.pose
        bar_pos = bar_pose.p
        tcp_pos = agent.tcp.pose.p

        depressions = torch.stack([self._pad_depression(pad) for pad in self.pads], dim=0)
        in_band = (depressions >= self._depression_min) & (depressions <= self._depression_max)
        both_loaded = in_band.all(dim=0)
        delta = torch.abs(depressions[0] - depressions[1])
        even = delta <= self._even_delta_tol

        bar_inv = bar_pose.inv()
        pad_top_local = torch.zeros((self.num_envs, 3), device=self.device)
        pad_top_local[:, 2] = self._pad_half_thickness
        seated_gaps = []
        cross_offsets = []
        overhangs = []
        for pad in self.pads:
            top_world = (pad.pose * Pose.create_from_pq(pad_top_local)).p
            in_bar = (bar_inv * Pose.create_from_pq(top_world)).p
            seated_gaps.append(torch.abs(in_bar[:, 2] + self._bar_half_thickness))
            cross_offsets.append(torch.abs(in_bar[:, 0]))
            overhangs.append(self._bar_half_length - torch.abs(in_bar[:, 1]))
        seated_gap = torch.stack(seated_gaps, dim=0)
        cross_offset = torch.stack(cross_offsets, dim=0)
        seated = ((seated_gap <= self._seated_z_tol) & (cross_offset <= self._bar_half_width)).all(dim=0)

        bar_cross = self._local_axis(bar_pose, 0)
        level = torch.abs(bar_cross[:, 2]) <= self._bar_roll_sin_tol

        grasp_width = EVEN_LOAD_BAR_SPEC.grasp_width
        assert grasp_width is not None
        finger_width = self.gripper_aperture()
        bar_axis = self._local_axis(bar_pose, 1)
        along = ((tcp_pos - bar_pos) * bar_axis).sum(dim=-1).clamp(-self._bar_half_length, self._bar_half_length)
        tcp_to_axis = torch.linalg.norm(tcp_pos - (bar_pos + along.unsqueeze(-1) * bar_axis), dim=-1)
        hands_off = (finger_width > grasp_width + self._hands_off_finger_margin) & (
            tcp_to_axis > self._hands_off_distance
        )

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        tilt_cos = self._local_axis(carrier_pose, 2)[:, 2]
        slip = torch.abs(self.carrier.linear_velocity[:, 1] - self.belt.velocity)
        slip_tol = self._carrier_slip_tol_base + self._carrier_slip_tol_rate * tolerance_speed(self.belt.velocity)
        carrier_riding = (
            (carrier_pos[:, 2] > deck_z)
            & (torch.abs(carrier_pos[:, 0] - ox) < self._conveyor_half_width + self._carrier_half_x)
            & (torch.abs(carrier_pos[:, 1] - oy) < self._conveyor_half_length + self._carrier_half_y)
            & (tilt_cos >= self._carrier_tilt_cos)
            & (slip <= slip_tol)
        )

        laid = both_loaded & even & seated & level & hands_off & carrier_riding

        self._even_sustain_counter = self._advance_sustain("even_sustain", self._even_sustain_counter, laid)
        success = self._even_sustain_counter >= self._even_sustain_steps

        bar_in_carrier = (carrier_pose.inv() * Pose.create_from_pq(bar_pos)).p

        return {
            "success": success,
            "both_loaded": both_loaded,
            "even": even,
            "seated": seated,
            "level": level,
            "hands_off": hands_off,
            "carrier_riding": carrier_riding,
            "depression_a": depressions[0],
            "depression_b": depressions[1],
            "depression_delta": delta,
            "bar_axis_offset": bar_in_carrier[:, 1],
            "bar_roll_sin": torch.abs(bar_cross[:, 2]),
            "bar_cross_offset": cross_offset.max(dim=0).values,
            "bar_overhang_min": torch.stack(overhangs, dim=0).min(dim=0).values,
            "seated_gap": seated_gap.max(dim=0).values,
            "tcp_to_bar_axis": tcp_to_axis,
            "carrier_slip": slip,
            "even_sustain": self._even_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        bar_pos = self.bar.pose.p
        carrier_pos = self.carrier.pose.p

        xy_dist = torch.linalg.norm(bar_pos[:, :2] - carrier_pos[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(8 * xy_dist)

        split_reward = (1.0 - info["depression_delta"] / self._even_delta_tol).clamp(0.0, 1.0)

        return carry_reward + 2.0 * split_reward * info["seated"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
