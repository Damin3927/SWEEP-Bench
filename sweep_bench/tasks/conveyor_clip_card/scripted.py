from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    _ScriptedInterceptMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedClipCardMixin(_ScriptedInterceptMixin):
    _PHASE_APPROACH = 0
    _PHASE_DESCEND = 1
    _PHASE_SEAT = 2
    _PHASE_RELEASE = 3
    _PHASE_WITHDRAW = 4

    carrier: Actor
    shim: Actor
    _tab_half_x: float
    _shim_half_x: float
    shim_half_thickness: float
    _tab_center_x: float
    _tab_half_height: float
    _carrier_deck_local_z: float
    _clip_knee_x: float
    _lip_run: float
    _clip_depth_target: float
    _release_finger_gap: float

    if TYPE_CHECKING:

        def _clip_seat_depth(self) -> torch.Tensor:
            pass

    _seat_z_offset: float = 0.0

    _scripted_hold_margin: float = 0.002

    _scripted_pos_step: float = 0.015
    _scripted_v_ee_max: float = 0.225

    _scripted_v_align: float = 0.15
    _scripted_max_lead: float = 0.30
    _scripted_comove_lead_time: float = 0.10

    _standoff_gap: float = 0.008

    _approach_z_above: float = 0.018

    _latch_u_tol: float = 0.014
    _latch_lane_tol: float = 0.008
    _latch_z_tol: float = 0.004
    _descend_u_tol: float = 0.030

    _max_seat_step: float = 0.022

    _release_depth_trigger: float = 0.028
    _seat_dwell_steps: int = 2
    _release_steps: int = 5
    _withdraw_z_above: float = 0.28

    @property
    def _scripted_rider(self) -> Actor:
        return self.carrier

    @property
    def _tab_grip_above_plate(self) -> float:
        return self.shim_half_thickness + self._tab_half_height

    @property
    def _lead_ahead_of_tcp(self) -> float:
        return self._shim_half_x - self._tab_center_x

    @property
    def _u_stand(self) -> float:
        return self._clip_knee_x - self._lip_run - self._standoff_gap - self._lead_ahead_of_tcp

    @property
    def _u_seat(self) -> float:
        return self._clip_knee_x + self._clip_depth_target - self._lead_ahead_of_tcp

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = rider_pos[:, 2] + self._carrier_deck_local_z + self.shim_half_thickness + self._tab_grip_above_plate
        return z, z

    def _hold_height(self, carrier_pos: torch.Tensor) -> torch.Tensor:
        return (
            carrier_pos[:, 2]
            + self._carrier_deck_local_z
            + self.shim_half_thickness
            + self._tab_grip_above_plate
            + self._seat_z_offset
        )

    def _scripted_commit_window(self) -> tuple[float, float]:
        return self._latch_u_tol, 2.0 * self._latch_u_tol

    def _scripted_hold_gripper(self) -> torch.Tensor:
        return self._gripper_action(self._tab_half_x - self._scripted_hold_margin)

    def _gripper_action(self, finger_target: float) -> torch.Tensor:
        value = self._gripper_action_for_aperture(2.0 * finger_target)
        return torch.full((self.num_envs, 1), value, device=self.device, dtype=torch.float32)

    def _gripper_down_target_q(self) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

        base_q = super()._gripper_down_target_q()
        half = -0.25 * math.pi * self.belt.direction_sign
        q_yaw = torch.zeros_like(base_q)
        q_yaw[:, 0] = torch.cos(half)
        q_yaw[:, 3] = torch.sin(half)
        return quaternion_multiply(q_yaw, base_q)

    def _clamp_pos_delta(self, delta_pos: torch.Tensor) -> torch.Tensor:
        from sweep_bench.tasks.conveyor_base.scripted import _as_col

        accel_step = _as_col(self._auto_scaled(self._scripted_accel_step))
        if accel_step is not None:
            prev = self._scripted_prev_delta
            accel = torch.as_tensor(accel_step, dtype=delta_pos.dtype, device=delta_pos.device)
            delta_pos = prev + (delta_pos - prev).clamp(-accel, accel)
        pos_step = torch.as_tensor(
            _as_col(self._auto_scaled(self._scripted_pos_step)),
            dtype=delta_pos.dtype,
            device=delta_pos.device,
        )
        cap = pos_step * self._advance_velrand_step() if self.velocity_randomization.enabled else pos_step
        clamped = torch.clamp(delta_pos, -cap, cap)
        self._scripted_prev_delta = clamped
        return clamped

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        b, device = self.num_envs, self.device
        if not hasattr(self, "_clip_phase"):
            self._clip_phase = torch.full((b,), self._PHASE_APPROACH, dtype=torch.long, device=device)
            self._clip_dwell_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._clip_release_counter = torch.zeros((b,), dtype=torch.long, device=device)
        self._reset_scripted_shared(env_idx)
        if env_idx is None:
            self._clip_phase[:] = self._PHASE_APPROACH
            self._clip_dwell_counter[:] = 0
            self._clip_release_counter[:] = 0
        else:
            self._clip_phase[env_idx] = self._PHASE_APPROACH
            self._clip_dwell_counter[env_idx] = 0
            self._clip_release_counter[env_idx] = 0

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_clip_phase"):
            self.reset_scripted()

        device = self.device
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q

        carrier_pos = self.carrier.pose.p
        sign = self.belt.direction_sign
        speed = torch.abs(self.belt.velocity)
        motion_dir = torch.zeros((self.num_envs, 2), device=device)
        motion_dir[:, 1] = sign
        self._scripted_motion_dir = motion_dir

        lane_x = carrier_pos[:, 0]
        hold_z = self._hold_height(carrier_pos)
        rel_x = tcp_pos[:, 0] - lane_x
        rel_z = tcp_pos[:, 2] - hold_z
        rel_z_high = rel_z - self._approach_z_above

        u = -(tcp_pos[:, 1] - carrier_pos[:, 1]) * sign

        phase = self._clip_phase
        at_station = ((u - self._u_stand).abs() < self._latch_u_tol) & (rel_x.abs() < self._latch_lane_tol)
        phase = torch.where(
            (phase == self._PHASE_APPROACH) & at_station & (rel_z_high.abs() < self._latch_z_tol),
            torch.full_like(phase, self._PHASE_DESCEND),
            phase,
        )
        on_deck = (rel_z.abs() < self._latch_z_tol) & ((u - self._u_stand).abs() < self._descend_u_tol)
        phase = torch.where((phase == self._PHASE_DESCEND) & on_deck, torch.full_like(phase, self._PHASE_SEAT), phase)

        depth = self._clip_seat_depth()
        seated = (phase == self._PHASE_SEAT) & (depth > self._release_depth_trigger)
        self._clip_dwell_counter = torch.where(
            seated, self._clip_dwell_counter + 1, torch.zeros_like(self._clip_dwell_counter)
        )
        phase = torch.where(
            seated & (self._clip_dwell_counter >= self._seat_dwell_steps),
            torch.full_like(phase, self._PHASE_RELEASE),
            phase,
        )

        in_release = phase == self._PHASE_RELEASE
        self._clip_release_counter = torch.where(in_release, self._clip_release_counter + 1, self._clip_release_counter)
        phase = torch.where(
            in_release & (self._clip_release_counter >= self._release_steps),
            torch.full_like(phase, self._PHASE_WITHDRAW),
            phase,
        )
        self._clip_phase = phase

        u_target = torch.where(
            phase <= self._PHASE_DESCEND,
            torch.full_like(u, self._u_stand),
            torch.full_like(u, self._u_seat),
        )

        ferry = phase == self._PHASE_APPROACH
        target_z = torch.where(ferry, hold_z + self._approach_z_above, hold_z)
        station = torch.stack([lane_x, carrier_pos[:, 1] - sign * u_target, target_z], dim=-1)
        self._update_auto_speed_scale(tcp_pos, station)

        z_gap = torch.where(ferry, rel_z_high, rel_z)
        align_gap = rel_x.abs() + self._scripted_lead_z_weight * z_gap.abs()
        v_align = self._auto_scaled(self._scripted_v_align)
        max_lead = self._auto_scaled(self._scripted_max_lead)
        lead = speed * self._scripted_comove_lead_time + speed * align_gap / v_align
        if self._scripted_auto_scale is not None:
            along = u_target - u
            near = (along.abs() < self._scripted_auto_bias_radius) & (rel_x.abs() < self._scripted_cross_scale)
            bias = torch.where(
                near,
                (self._scripted_auto_lead_bias - self._scripted_auto_bias_gain * along).clamp(
                    0.0, self._scripted_auto_bias_max
                ),
                self._scripted_auto_lead_bias,
            )
            self._scripted_auto_lead_bias = bias
            lead = lead + bias
        lead = lead.clamp(max=max_lead)

        err = u_target - u
        cap = lead + self._max_seat_step
        du = torch.where(phase >= self._PHASE_SEAT, torch.clamp(err, -cap, cap), err)
        u_cmd = u + du

        aim_y = carrier_pos[:, 1] - sign * u_cmd + sign * lead
        target_world = torch.stack([lane_x, aim_y, target_z], dim=-1)

        is_withdraw = phase == self._PHASE_WITHDRAW
        target_world[:, 2] = torch.where(
            is_withdraw,
            torch.full_like(target_world[:, 2], deck_z + self._withdraw_z_above),
            target_world[:, 2],
        )

        pos_input = self._position_input(target_world, tcp_pos)
        rot_input = self._rotation_input(tcp_q)

        gripper = torch.where(
            (is_withdraw | in_release).unsqueeze(-1),
            self._gripper_action(self._release_finger_gap),
            self._scripted_hold_gripper(),
        )
        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
