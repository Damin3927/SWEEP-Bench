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
from sweep_bench.tasks.conveyor_base.on_belt import carried_speed
from sweep_bench.tasks.conveyor_clock_hand.assets import build_clock_carrier, build_clock_hand
from sweep_bench.tasks.conveyor_clock_hand.config import RandomizationConfig
from sweep_bench.tasks.conveyor_clock_hand.scripted import ScriptedClockHandMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor

_TWO_PI = 2.0 * math.pi


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, _TWO_PI) - math.pi


@bench_task("ConveyorClockHand-v0")
@register_env("ConveyorClockHand-v0", max_episode_steps=300)
class ConveyorClockHandEnv(ScriptedClockHandMixin, ConveyorWorkspaceEnv):
    """Nudge a free gauge needle to a commanded angle BAND on a moving dial."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _disc_radius: float = 0.045
    _disc_half_thickness: float = 0.004
    _boss_half_width: float = 0.028
    _well_half_width: float = 0.014
    _well_depth: float = 0.0045
    _ledge_thickness: float = 0.003
    _pin_clearance: float = 0.0005
    _carrier_density: float = 2000.0

    _collar_radius: float = 0.012
    _collar_half_thickness: float = 0.002
    _pin_radius: float = 0.004
    _pin_half_length: float = 0.0025
    _plate_half_width: float = 0.004
    _plate_half_length: float = 0.040
    _plate_half_thickness: float = 0.002
    _pad_radius: float = 0.036
    _pad_half_width: float = 0.005
    _pad_half_height: float = 0.014
    _contact_above_pad_bottom: float = 0.009
    _hand_density: float = 2000.0
    _journal_friction: float = 0.08

    _carrier_patch_radius: float = 0.030

    _finger_below_tcp: float = 0.009

    _hub_viscous: float = 1.0e-3
    _hub_coulomb: float = 4.0e-6
    _hub_coulomb_knee: float = 0.02
    _hub_torque_max: float = 2.0e-3

    _target_deg_range: tuple[int, int] = (20, 60)

    _clock_band: float = math.radians(12.0)
    _clock_max_ang_speed: float = 0.5
    _hub_lateral_tol: float = 0.002
    _hub_axial_tol: float = 0.002
    _clock_tcp_clear: float = 0.05
    _clock_deck_drop: float = 0.005
    _clock_on_belt_y: float = 0.85
    _clock_track_frac: float = 0.5
    _clock_sustain_steps: int = 8

    default_prompt: str = "turn the gauge needle to the commanded angle on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _aperture_half_width(self) -> float:
        return self._pin_radius + self._pin_clearance

    @property
    def _hub_rest_z(self) -> float:
        return 2.0 * self._disc_half_thickness + self._collar_half_thickness

    @property
    def _pad_local_z(self) -> float:
        plate_z = self._collar_half_thickness + 2.0 * self._pin_half_length + self._plate_half_thickness
        return plate_z + self._plate_half_thickness + self._pad_half_height

    @property
    def _pad_bottom_local_z(self) -> float:
        return self._hub_rest_z + self._pad_local_z - self._pad_half_height

    @property
    def _needle_half_span(self) -> float:
        return max(self._plate_half_length, self._pad_radius + self._pad_half_width)

    @property
    def _contact_local_z(self) -> float:
        return self._pad_bottom_local_z + self._contact_above_pad_bottom + self._finger_below_tcp

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        carrier_builder = build_clock_carrier(
            self.scene,
            disc_radius=self._disc_radius,
            disc_half_thickness=self._disc_half_thickness,
            boss_half_width=self._boss_half_width,
            well_half_width=self._well_half_width,
            well_depth=self._well_depth,
            ledge_thickness=self._ledge_thickness,
            aperture_half_width=self._aperture_half_width,
            density=self._carrier_density,
        )
        carrier_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z])
        self.carrier = carrier_builder.build(name="clock_carrier")
        self.belt.register_rider(self.carrier, patch_radius=self._carrier_patch_radius)

        hand_builder = build_clock_hand(
            self.scene,
            collar_radius=self._collar_radius,
            collar_half_thickness=self._collar_half_thickness,
            pin_radius=self._pin_radius,
            pin_half_length=self._pin_half_length,
            plate_half_width=self._plate_half_width,
            plate_half_length=self._plate_half_length,
            plate_half_thickness=self._plate_half_thickness,
            pad_radius=self._pad_radius,
            pad_half_width=self._pad_half_width,
            pad_half_height=self._pad_half_height,
            density=self._hand_density,
            friction=self._journal_friction,
        )
        hand_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._hub_rest_z])
        self.hand = hand_builder.build(name="clock_hand")

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.hand]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        carrier_q = self._sample_carrier_orientation(env_idx, b)
        carrier_pos = self._sample_carrier_position(env_idx, b)
        self.carrier.set_pose(Pose.create_from_pq(carrier_pos, carrier_q))
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        self.carrier.set_linear_velocity(vel)
        self.carrier.set_angular_velocity(torch.zeros((b, 3), device=device))

        start = self._sample_hand_start_angle(b)
        hub_local = torch.zeros((b, 3), device=device)
        hub_local[:, 2] = self._hub_rest_z
        rot = quaternion_to_matrix(carrier_q)
        hand_pos = carrier_pos + torch.bmm(rot, hub_local.unsqueeze(-1)).squeeze(-1)
        spin_q = torch.zeros((b, 4), device=device)
        spin_q[:, 0] = torch.cos(start * 0.5)
        spin_q[:, 3] = torch.sin(start * 0.5)
        self.hand.set_pose(Pose.create_from_pq(hand_pos, quaternion_multiply(carrier_q, spin_q)))
        self.hand.set_linear_velocity(vel)
        self.hand.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_clock_trackers(env_idx)
        deg = self._episode_rng.randint(self._target_deg_range[0], self._target_deg_range[1] + 1, size=b)
        sign = self._episode_rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=b)
        self._clock_target_deg[env_idx] = torch.as_tensor(deg.astype(np.float32) * sign, device=device)
        self._clock_sign[env_idx] = torch.as_tensor(sign, device=device)
        self._clock_target[env_idx] = start + torch.as_tensor(np.radians(deg.astype(np.float32)) * sign, device=device)
        self._hub_viscous_env[env_idx], self._hub_coulomb_env[env_idx] = self._sample_hub_damping(b)
        self._hub_damping_ready = True

    def _sample_carrier_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        yaw_np = self._episode_rng.uniform(*self.randomization.yaw_range, size=b).astype(np.float32)
        yaw = (
            torch.as_tensor(yaw_np, device=device) if self.randomization.carrier_yaw else torch.zeros(b, device=device)
        )
        yaw = yaw + torch.where(
            self.belt.direction_sign[env_idx] < 0.0, torch.full_like(yaw, math.pi), torch.zeros_like(yaw)
        )
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = torch.cos(yaw * 0.5)
        q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_hand_start_angle(self, b: int) -> torch.Tensor:
        if not self.randomization.hand_start_angle:
            return torch.zeros(b, device=self.device)
        lo, hi = self.randomization.start_angle_range
        return torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)

    def _sample_hub_damping(self, b: int) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.ones(b, device=self.device)
        if self.randomization.hub_damping:
            lo, hi = self.randomization.hub_damping_range
            scale = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)
        return scale * self._hub_viscous, scale * self._hub_coulomb

    def _sample_carrier_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._disc_radius - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.rider_lateral_offset:
            lo, hi = self.randomization.lateral_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = pos[:, 0] + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z
        return pos

    @property
    def episode_prompts(self) -> list[str]:
        return [
            f"turn the gauge needle {abs(int(d))} degrees "
            f"{'counter-clockwise' if d > 0 else 'clockwise'} on the conveyor belt"
            for d in self._clock_target_deg.tolist()
        ]

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_hub_damping()

    def _apply_hub_damping(self) -> None:
        if not getattr(self, "_hub_damping_ready", False):
            return
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()
        axis = quaternion_to_matrix(self.carrier.pose.q)[:, :, 2]
        rel_omega = self._clock_hand_spin()
        tau = -(
            self._hub_viscous_env * rel_omega + self._hub_coulomb_env * torch.tanh(rel_omega / self._hub_coulomb_knee)
        ).clamp(-self._hub_torque_max, self._hub_torque_max)
        torque = axis * tau.unsqueeze(-1)
        if gpu:
            torque_buf = self.scene.px.cuda_rigid_body_torque.torch()
            torque_buf[self.hand._body_data_index, :3] = torque
            self.scene.px.gpu_apply_rigid_dynamic_torque()
        else:
            self.hand._bodies[0].add_force_torque(np.zeros(3, dtype=np.float32), torque.squeeze(0).cpu().numpy())

    def _clock_hand_angle(self) -> torch.Tensor:
        rot_c = quaternion_to_matrix(self.carrier.pose.q)
        rot_h = quaternion_to_matrix(self.hand.pose.q)
        rel = torch.bmm(rot_c.transpose(1, 2), rot_h)
        return torch.atan2(rel[:, 1, 0], rel[:, 0, 0])

    def _clock_hand_spin(self) -> torch.Tensor:
        axis = quaternion_to_matrix(self.carrier.pose.q)[:, :, 2]
        return ((self.hand.angular_velocity - self.carrier.angular_velocity) * axis).sum(dim=-1)

    def _hub_offsets(self) -> tuple[torch.Tensor, torch.Tensor]:
        rot_c = quaternion_to_matrix(self.carrier.pose.q)
        hub = torch.bmm(rot_c.transpose(1, 2), (self.hand.pose.p - self.carrier.pose.p).unsqueeze(-1)).squeeze(-1)
        return torch.linalg.norm(hub[:, :2], dim=-1), torch.abs(hub[:, 2] - self._hub_rest_z)

    def _pointer_tip(self) -> torch.Tensor:
        rot_h = quaternion_to_matrix(self.hand.pose.q)
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 1] = self._pad_radius
        local[:, 2] = self._pad_local_z
        return self.hand.pose.p + torch.bmm(rot_h, local.unsqueeze(-1)).squeeze(-1)

    def _needle_clearance(self) -> torch.Tensor:
        agent = cast("Panda", self.agent)
        rot_h = quaternion_to_matrix(self.hand.pose.q)
        local = torch.bmm(rot_h.transpose(1, 2), (agent.tcp.pose.p - self.hand.pose.p).unsqueeze(-1)).squeeze(-1)
        nearest = torch.zeros_like(local)
        nearest[:, 1] = local[:, 1].clamp(-self._needle_half_span, self._needle_half_span)
        nearest[:, 2] = self._pad_local_z
        return torch.linalg.norm(local - nearest, dim=-1)

    @property
    def _clock_target_angle(self) -> torch.Tensor:
        return self._clock_target

    @property
    def _clock_move_sign(self) -> torch.Tensor:
        return self._clock_sign

    def _reset_clock_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_clock_sustain_counter"):
            b, device = self.num_envs, self.device
            self._clock_sustain_counter = torch.zeros(b, dtype=torch.long, device=device)
            self._clock_target = torch.zeros(b, device=device)
            self._clock_target_deg = torch.zeros(b, device=device)
            self._clock_sign = torch.ones(b, device=device)
            self._hub_viscous_env = torch.full((b,), self._hub_viscous, device=device)
            self._hub_coulomb_env = torch.full((b,), self._hub_coulomb, device=device)
        self._clock_sustain_counter[env_idx] = 0

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        angle = self._clock_hand_angle()
        angle_err = _wrap(angle - self._clock_target)
        carrier_pos = self.carrier.pose.p

        in_band = torch.abs(angle_err) <= self._clock_band

        spin = self._clock_hand_spin()
        settled = torch.abs(spin) < self._clock_max_ang_speed

        hub_lateral, hub_axial = self._hub_offsets()
        on_pivot = (hub_lateral < self._hub_lateral_tol) & (hub_axial < self._hub_axial_tol)

        needle_dist = self._needle_clearance()
        arm_clear = needle_dist >= self._clock_tcp_clear

        on_belt = (
            (carrier_pos[:, 2] > deck_z - self._clock_deck_drop)
            & (torch.abs(carrier_pos[:, 0] - ox) < self._conveyor_half_width)
            & (torch.abs(carrier_pos[:, 1] - oy) < self._clock_on_belt_y)
        )

        belt_speed = torch.abs(self.belt.velocity)
        travelling = torch.abs(self.carrier.linear_velocity[:, 1]) >= carried_speed(belt_speed, self._clock_track_frac)

        holding = in_band & settled & on_pivot & arm_clear & on_belt & travelling
        self._clock_sustain_counter = self._advance_sustain("clock_sustain", self._clock_sustain_counter, holding)
        success = self._clock_sustain_counter >= self._clock_sustain_steps

        return {
            "success": success,
            "hand_angle": angle,
            "angle_error": angle_err,
            "target_angle": self._clock_target,
            "in_band": in_band,
            "needle_settled": settled,
            "hub_lateral": hub_lateral,
            "hub_axial": hub_axial,
            "on_pivot": on_pivot,
            "tcp_to_needle": needle_dist,
            "arm_clear": arm_clear,
            "carrier_on_belt": on_belt,
            "carrier_travelling": travelling,
            "clock_sustain": self._clock_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p

        tcp_to_pad = torch.linalg.norm(tcp_pos - self._pointer_tip(), dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_pad)

        set_reward = 1 - torch.tanh(3 * torch.abs(info["angle_error"]))

        return reach_reward + 2.0 * set_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
