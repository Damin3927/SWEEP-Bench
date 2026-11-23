from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import tolerance_speed
from sweep_bench.tasks.conveyor_clip_card.assets import (
    build_clip_carrier,
    build_clip_pad,
    build_shim,
)
from sweep_bench.tasks.conveyor_clip_card.config import RandomizationConfig
from sweep_bench.tasks.conveyor_clip_card.scripted import ScriptedClipCardMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorClipCard-v0")
@register_env("ConveyorClipCard-v0", max_episode_steps=300)
class ConveyorClipCardEnv(ScriptedClipCardMixin, ConveyorWorkspaceEnv):
    """Slide a held shim under a sprung clip so the clip's own preload holds it."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _carrier_half_x: float = 0.115
    _carrier_half_y: float = 0.048
    _carrier_deck_half_thickness: float = 0.010
    _pin_x: float = 0.105
    _pin_height: float = 0.006
    _pin_radius: float = 0.0035
    _pin_clearance: float = 0.0006
    _journal_inner_y: float = 0.032
    _journal_outer_y: float = 0.043
    _wall_half_thickness: float = 0.002
    _carrier_density: float = 3600.0
    _carrier_patch_radius: float = 0.035
    _bearing_friction: float = 0.05

    _plate_half_thickness: float = 0.002
    _plate_root_gap: float = 0.005
    _pin_half_length: float = 0.0455
    _collar_radius: float = 0.006
    _collar_half_length: float = 0.0015
    _collar_y: float = 0.0445
    _pad_density: float = 1200.0
    _pad_angular_damping: float = 3.0
    _clip_return_force: float = 1.0

    _shim_half_x: float = 0.045
    _shim_half_y: float = 0.025
    _tab_half_x: float = 0.005
    _tab_half_y: float = 0.014
    _tab_half_height: float = 0.011
    _tab_center_x: float = -0.032
    _shim_density: float = 500.0
    _release_finger_gap: float = 0.015
    _finger_body_clear: float = 0.008

    _clip_min_depth: float = 0.025
    _clip_lateral_tol: float = 0.008
    _clip_band_lo: float = 0.004
    _clip_band_hi: float = 0.010
    _clip_hands_off_dist: float = 0.09
    _clip_hands_off_z: float = 0.030
    _clip_open_margin: float = 0.010
    _clip_drift_tol: float = 0.006
    _clip_pin_capture: float = 0.010
    _carrier_drift_tol: float = 0.008
    _carrier_lift_tol: float = 0.006
    _clip_speed_tol_base: float = 0.06
    _clip_speed_tol_per_speed: float = 0.20
    _clip_sustain_steps: int = 8

    default_prompt: str = "slide the shim under the clip on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        pad_length: float = 0.150,
        plate_half_y: float = 0.030,
        lip_half_length: float = 0.010,
        lip_angle: float = math.radians(35.0),
        shim_half_thickness: float = 0.003,
        **kwargs,
    ) -> None:
        self.pad_length: float = pad_length
        self.plate_half_y: float = plate_half_y
        self.lip_half_length: float = lip_half_length
        self.lip_angle: float = lip_angle
        self.shim_half_thickness: float = shim_half_thickness
        self._validate_clip_geometry()
        super().__init__(*args, **kwargs)

    def _validate_clip_geometry(self) -> None:
        thickness = 2.0 * self.shim_half_thickness
        rise = 2.0 * self.lip_half_length * math.sin(self.lip_angle)
        if rise <= thickness:
            raise ValueError(
                f"lip rise {rise:.4f} m must exceed the shim thickness {thickness:.4f} m, or the shim's flat "
                "leading face meets the lip head-on (a horizontal contact normal) and cannot open the clip"
            )
        if self._clip_band_lo >= thickness:
            raise ValueError(
                f"preload band floor {self._clip_band_lo} must sit below the shim thickness {thickness:.4f}, "
                "or a correctly pinched shim reads as 'the clip shut on nothing'"
            )
        depth_max = self.pad_length * (1.0 - thickness / self._clip_band_hi)
        if depth_max <= self._clip_min_depth:
            raise ValueError(
                f"pad_length={self.pad_length} with a {thickness:.4f} m shim caps the legal depth at "
                f"{depth_max:.4f} m, at or below the {self._clip_min_depth} m the depth term demands: "
                "the preload band and the depth term cannot both be satisfied (lengthen the pad)"
            )
        if not self._clip_min_depth < self._clip_depth_target < depth_max:
            raise ValueError(
                f"commanded seat depth {self._clip_depth_target} is outside the legal window "
                f"({self._clip_min_depth}, {depth_max:.4f})"
            )
        tab_x = self._clip_knee_x + self._clip_depth_target - self._shim_half_x + self._tab_center_x
        finger_face_x = tab_x + self._release_finger_gap
        clearance = (self._clip_knee_x - self._lip_run) - finger_face_x
        if clearance < self._finger_body_clear:
            raise ValueError(
                f"the opened fingers clear the lip tip by only {clearance:.4f} m at the commanded seat depth; "
                f"they need {self._finger_body_clear} m (move `_tab_center_x` back or lengthen the shim)"
            )
        if not -self._shim_half_x < self._tab_center_x - self._tab_half_x:
            raise ValueError("the grip tab overhangs the shim's trailing edge")
        shim_center_x = self._clip_knee_x + self._clip_depth_target - self._shim_half_x
        if shim_center_x < -self._carrier_half_x:
            raise ValueError(
                f"the seated shim's centre ({shim_center_x:.4f}) hangs off the carrier deck "
                f"(half-extent {self._carrier_half_x}); lengthen the carrier"
            )

    @property
    def _carrier_deck_local_z(self) -> float:
        return 2.0 * self._carrier_deck_half_thickness

    @property
    def _clip_knee_x(self) -> float:
        return self._pin_x - self.pad_length

    @property
    def _lip_run(self) -> float:
        return 2.0 * self.lip_half_length * math.cos(self.lip_angle)

    @property
    def _clip_depth_target(self) -> float:
        return 0.032

    def _free_edge_lift(self, depth: float) -> float:
        return 2.0 * self.shim_half_thickness * self.pad_length / (self.pad_length - depth)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        carrier_builder = build_clip_carrier(
            self.scene,
            half_x=self._carrier_half_x,
            half_y=self._carrier_half_y,
            deck_half_thickness=self._carrier_deck_half_thickness,
            pin_x=self._pin_x,
            pin_height=self._pin_height,
            pin_radius=self._pin_radius,
            pin_clearance=self._pin_clearance,
            journal_inner_y=self._journal_inner_y,
            journal_outer_y=self._journal_outer_y,
            wall_half_thickness=self._wall_half_thickness,
            density=self._carrier_density,
            bearing_friction=self._bearing_friction,
        )
        carrier_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z])
        self.carrier = carrier_builder.build(name="clip_carrier")
        self.belt.register_rider(self.carrier, patch_radius=self._carrier_patch_radius)

        pad_builder = build_clip_pad(
            self.scene,
            pad_length=self.pad_length,
            plate_half_y=self.plate_half_y,
            plate_half_thickness=self._plate_half_thickness,
            plate_root_gap=self._plate_root_gap,
            pin_height=self._pin_height,
            lip_half_length=self.lip_half_length,
            lip_angle=self.lip_angle,
            pin_radius=self._pin_radius,
            pin_half_length=self._pin_half_length,
            collar_radius=self._collar_radius,
            collar_half_length=self._collar_half_length,
            collar_y=self._collar_y,
            density=self._pad_density,
            bearing_friction=self._bearing_friction,
        )
        pad_builder.initial_pose = sapien.Pose(
            p=[ox + self._pin_x, oy, deck_z + self._carrier_deck_local_z + self._pin_height]
        )
        self.pad = pad_builder.build(name="clip_pad")
        for body in self.pad._bodies:
            body.angular_damping = self._pad_angular_damping

        shim_builder = build_shim(
            self.scene,
            half_x=self._shim_half_x,
            half_y=self._shim_half_y,
            half_thickness=self.shim_half_thickness,
            tab_half_x=self._tab_half_x,
            tab_half_y=self._tab_half_y,
            tab_half_height=self._tab_half_height,
            tab_center_x=self._tab_center_x,
            density=self._shim_density,
        )
        shim_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.shim = shim_builder.build(name="clip_shim")

        lead = torch.zeros((self.num_envs, 3), device=self.device)
        lead[:, 0] = self._shim_half_x
        self._shim_lead_offset = Pose.create_from_pq(p=lead)
        knee = torch.zeros((self.num_envs, 3), device=self.device)
        knee[:, 0] = -self.pad_length
        knee[:, 2] = -self._pin_height
        self._pad_knee_offset = Pose.create_from_pq(p=knee)
        self._pin_anchor = torch.tensor(
            [self._pin_x, 0.0, self._carrier_deck_local_z + self._pin_height], device=self.device
        )

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._tab_half_x - self._scripted_hold_margin, dtype=np.float32)

    def _carrier_spawn_q(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        q = torch.zeros((b, 4), device=self.device)
        half = -0.25 * math.pi * self.belt.direction_sign[env_idx]
        q[:, 0] = torch.cos(half)
        q[:, 3] = torch.sin(half)
        return q

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        yaw_q = self._carrier_spawn_q(env_idx, b)
        sign = self.belt.direction_sign[env_idx]

        carrier_pos = self._sample_carrier_position(env_idx, b)
        self.carrier.set_pose(Pose.create_from_pq(carrier_pos, yaw_q))
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        self.carrier.set_linear_velocity(vel)
        self.carrier.set_angular_velocity(torch.zeros((b, 3), device=device))

        pad_pos = carrier_pos.clone()
        pad_pos[:, 1] = pad_pos[:, 1] - self._pin_x * sign
        pad_pos[:, 2] = pad_pos[:, 2] + self._carrier_deck_local_z + self._pin_height
        self.pad.set_pose(Pose.create_from_pq(pad_pos, yaw_q.clone()))
        self.pad.set_linear_velocity(vel.clone())
        self.pad.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pos = cast("Panda", self.agent).tcp.pose.p[env_idx]
        shim_pos = tcp_pos.clone()
        shim_pos[:, 1] = tcp_pos[:, 1] + self._tab_center_x * sign
        shim_pos[:, 2] = tcp_pos[:, 2] - self.shim_half_thickness - self._tab_half_height
        self.shim.set_pose(Pose.create_from_pq(shim_pos, yaw_q.clone()))
        self.shim.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.shim.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._sample_clip_return_force(env_idx, b)
        self._reset_clip_trackers(env_idx, carrier_pos)

    def _sample_carrier_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._carrier_half_x - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z
        return pos

    def _sample_clip_return_force(self, env_idx: torch.Tensor, b: int) -> None:
        if not hasattr(self, "_clip_force"):
            self._clip_force = torch.full((self.num_envs,), self._clip_return_force, device=self.device)
        if self.randomization.clip_torque:
            lo, hi = self.randomization.clip_force_range
            draw = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            self._clip_force[env_idx] = torch.as_tensor(draw, device=self.device)
        else:
            self._clip_force[env_idx] = self._clip_return_force

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_clip_return_torque()

    def _apply_clip_return_torque(self) -> None:
        if not hasattr(self, "_clip_force"):
            return
        from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

        hinge_axis = quaternion_to_matrix(self.pad.pose.q)[:, :, 1]
        torque = -(self._clip_force * self.pad_length).unsqueeze(-1) * hinge_axis
        force = torch.zeros((self.num_envs, 3), device=self.device)
        if self.scene.gpu_sim_enabled:
            px = self.scene.px
            px.cuda_rigid_body_force.torch()[self.pad._body_data_index, :3] = force
            px.cuda_rigid_body_torque.torch()[self.pad._body_data_index, :3] = torque
            px.gpu_apply_rigid_dynamic_force()
            px.gpu_apply_rigid_dynamic_torque()
        else:
            self.pad._bodies[0].add_force_torque(force.squeeze(0).cpu().numpy(), torque.squeeze(0).cpu().numpy())

    def _shim_in_carrier(self) -> torch.Tensor:
        return (self.carrier.pose.inv() * self.shim.pose).p

    def _clip_seat_depth(self) -> torch.Tensor:
        lead_in_carrier = (self.carrier.pose.inv() * (self.shim.pose * self._shim_lead_offset)).p
        return lead_in_carrier[:, 0] - self._clip_knee_x

    def _pad_free_edge_height(self) -> torch.Tensor:
        knee_in_carrier = (self.carrier.pose.inv() * (self.pad.pose * self._pad_knee_offset)).p
        return knee_in_carrier[:, 2] - self._carrier_deck_local_z

    def _pad_pin_offset(self) -> torch.Tensor:
        pin_in_carrier = (self.carrier.pose.inv() * self.pad.pose).p
        return torch.linalg.norm(pin_in_carrier - self._pin_anchor, dim=-1)

    def _reset_clip_trackers(self, env_idx: torch.Tensor, carrier_pos: torch.Tensor) -> None:
        if not hasattr(self, "_clip_sustain_counter"):
            self._clip_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._carrier_spawn_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._clip_ref_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._clip_sustain_counter[env_idx] = 0
        self._carrier_spawn_pos[env_idx] = carrier_pos
        self._clip_ref_pos[env_idx] = 0.0

    def evaluate(self):
        oy = self.workspace_offset[1]
        carrier_pos = self.carrier.pose.p
        shim_rel = self._shim_in_carrier()

        depth = self._clip_seat_depth()
        inserted = (depth > self._clip_min_depth) & (shim_rel[:, 1].abs() < self._clip_lateral_tol)

        lift = self._pad_free_edge_height()
        preloaded = (lift > self._clip_band_lo) & (lift < self._clip_band_hi)

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        finger_width = self.gripper_aperture()
        deck_top_z = carrier_pos[:, 2] + self._carrier_deck_local_z
        hands_off = (
            (finger_width >= 2.0 * self._tab_half_x + self._clip_open_margin)
            & (torch.linalg.norm(tcp_pos - self.shim.pose.p, dim=-1) > self._clip_hands_off_dist)
            & (tcp_pos[:, 2] > deck_top_z + self._clip_hands_off_z)
        )

        window_open = self._clip_sustain_counter == 0
        self._clip_ref_pos = torch.where(window_open.unsqueeze(-1), shim_rel, self._clip_ref_pos)
        drift = torch.linalg.norm(shim_rel - self._clip_ref_pos, dim=-1)
        retained = drift < self._clip_drift_tol

        pin_offset = self._pad_pin_offset()
        hinge_captured = pin_offset < self._clip_pin_capture

        spawn = self._carrier_spawn_pos
        drift_x = (carrier_pos[:, 0] - spawn[:, 0]).abs()
        speed_tol = self._clip_speed_tol_base + self._clip_speed_tol_per_speed * tolerance_speed(self.belt.velocity)
        v_slip = (self.carrier.linear_velocity[:, 1] - self.belt.velocity).abs()
        carrier_riding = (
            ((carrier_pos[:, 1] - oy).abs() < self._conveyor_half_length)
            & ((carrier_pos[:, 2] - spawn[:, 2]).abs() < self._carrier_lift_tol)
            & (drift_x < self._carrier_drift_tol)
            & (v_slip <= speed_tol)
        )

        holding = inserted & preloaded & hands_off & retained & hinge_captured & carrier_riding
        self._clip_sustain_counter = self._advance_sustain("clip_sustain", self._clip_sustain_counter, holding)
        success = self._clip_sustain_counter >= self._clip_sustain_steps

        return {
            "success": success,
            "inserted": inserted,
            "preloaded": preloaded,
            "hands_off": hands_off,
            "retained": retained,
            "hinge_captured": hinge_captured,
            "carrier_riding": carrier_riding,
            "clip_depth": depth,
            "clip_lift": lift,
            "clip_drift": drift,
            "pin_offset": pin_offset,
            "carrier_drift_x": drift_x,
            "carrier_v_slip": v_slip,
            "carrier_band_margin": self._conveyor_half_width
            - self._carrier_half_y
            - (carrier_pos[:, 0] - self.workspace_offset[0]).abs(),
            "clip_sustain": self._clip_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p

        mouth_local = torch.zeros((self.num_envs, 3), device=self.device)
        mouth_local[:, 0] = self._clip_knee_x - self._lip_run
        mouth_local[:, 2] = self._carrier_deck_local_z
        mouth = (self.carrier.pose * Pose.create_from_pq(p=mouth_local)).p
        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - mouth, dim=-1))

        depth = (info["clip_depth"] / self._clip_depth_target).clamp(0.0, 1.0)
        aligned = (self._shim_in_carrier()[:, 1].abs() < self._clip_lateral_tol).float()

        return reach_reward + 2.0 * depth * aligned + 2.0 * (info["preloaded"] & info["hands_off"]).float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
