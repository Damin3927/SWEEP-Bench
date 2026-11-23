from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion

from sweep_bench.tasks.conveyor_base.scripted_families import (
    _ScriptedInterceptMixin,
    _smoothstep,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedSheetEdgePickMixin(_ScriptedInterceptMixin):
    _PHASE_APPROACH = 0
    _PHASE_CLOSE = 1
    _PHASE_LIFT = 2
    _PHASE_HOLD = 3

    sheet: Actor
    _sheet_half_x: float
    _rail_thickness: float
    _entry_aperture: float
    _approach_tilt_deg: float

    _scripted_pos_step: float = 0.015
    _scripted_v_ee_max: float = 0.225

    _scripted_v_align: float = 0.15
    _scripted_max_lead: float = 0.30

    _station_y_lo: float = -0.25
    _station_y_hi: float = 0.45

    _standoff_gap: float = 0.021
    _grip_inset: float = 0.017

    _commit_y_in: float = 0.020
    _commit_y_out: float = 0.055
    _commit_z_in: float = 0.004
    _commit_z_out: float = 0.012
    _commit_rot_lo: float = 0.985
    _commit_rot_hi: float = 0.998

    _max_lateral_step: float = 0.008
    _lateral_guard_gap: float = 0.010

    _close_x_tol: float = 0.004
    _close_y_tol: float = 0.045
    _close_z_tol: float = 0.006
    _close_rot_cos: float = 0.995
    _scripted_close_steps: int = 6

    _scripted_lift_z_above: float = 0.20
    _scripted_hold_z_above: float = 0.185

    @property
    def _scripted_rider(self) -> Actor:
        return self.sheet

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lip_z = rider_pos[:, 2] + 0.5 * self._rail_thickness
        return lip_z, lip_z

    def _scripted_commit_window(self) -> tuple[float, float]:
        return self._commit_y_in, self._commit_y_out

    def _gripper_down_target_q(self) -> torch.Tensor:
        device = self.device
        sign = self.belt.direction_sign
        tilt = math.radians(self._approach_tilt_deg)
        cos_t = torch.full_like(sign, math.cos(tilt))
        sin_t = torch.full_like(sign, math.sin(tilt))
        zero = torch.zeros_like(sign)

        x_body = torch.stack([sin_t, sign * cos_t, zero], dim=-1)
        y_body = torch.stack([zero, zero, sign], dim=-1)
        z_body = torch.stack([cos_t, -sign * sin_t, zero], dim=-1)
        R = torch.stack([x_body, y_body, z_body], dim=-1).to(device)
        return matrix_to_quaternion(R)

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
        if not hasattr(self, "_sep_phase"):
            self._sep_phase = torch.full((b,), self._PHASE_APPROACH, dtype=torch.long, device=device)
            self._sep_close_counter = torch.zeros((b,), dtype=torch.long, device=device)
        self._reset_scripted_shared(env_idx)

        if env_idx is None:
            self._sep_phase[:] = self._PHASE_APPROACH
            self._sep_close_counter[:] = 0
        else:
            self._sep_phase[env_idx] = self._PHASE_APPROACH
            self._sep_close_counter[env_idx] = 0

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_sep_phase"):
            self.reset_scripted()

        device = self.device
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        oy = self.workspace_offset[1]
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        tcp_vel = agent.tcp.linear_velocity

        sheet_pos = self.sheet.pose.p
        sign = self.belt.direction_sign
        speed = torch.abs(self.belt.velocity)
        motion_dir = torch.zeros((self.num_envs, 2), device=device)
        motion_dir[:, 1] = sign
        self._scripted_motion_dir = motion_dir

        lip_x = sheet_pos[:, 0] - self._sheet_half_x
        lip_z = sheet_pos[:, 2] + 0.5 * self._rail_thickness
        stand_x = lip_x - self._standoff_gap
        grip_x = lip_x + self._grip_inset

        station = torch.stack([stand_x, sheet_pos[:, 1], lip_z], dim=-1)
        self._update_auto_speed_scale(tcp_pos, station)

        phase = self._sep_phase

        rel_z = tcp_pos[:, 2] - lip_z
        along = (tcp_pos[:, 1] - sheet_pos[:, 1]) * sign
        approach_gap = (stand_x - tcp_pos[:, 0]).clamp(min=0.0)
        align_gap = approach_gap + self._scripted_lead_z_weight * rel_z.abs()
        v_align = self._auto_scaled(self._scripted_v_align)
        max_lead = self._auto_scaled(self._scripted_max_lead)
        lead = speed * self._scripted_comove_lead_time + speed * align_gap / v_align
        if self._scripted_auto_scale is not None:
            near = (along.abs() < self._scripted_auto_bias_radius) & (
                (tcp_pos[:, 0] - stand_x).abs() < self._scripted_cross_scale
            )
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
        aim_travel = ((sheet_pos[:, 1] - oy) * sign + lead).clamp(self._station_y_lo, self._station_y_hi)
        aim_y = oy + sign * aim_travel

        target_q = self._gripper_down_target_q()
        rot_cos = torch.abs((tcp_q * target_q).sum(dim=-1))
        s_y = 1.0 - _smoothstep((along.abs() - self._commit_y_in) / (self._commit_y_out - self._commit_y_in))
        s_z = 1.0 - _smoothstep((rel_z.abs() - self._commit_z_in) / (self._commit_z_out - self._commit_z_in))
        s_rot = _smoothstep((rot_cos - self._commit_rot_lo) / (self._commit_rot_hi - self._commit_rot_lo))
        commit = s_y * s_z * s_rot
        if self._scripted_auto_scale is not None:
            slip = torch.abs(tcp_vel[:, 1] * sign - speed)
            crossing = _smoothstep(
                (slip - self._scripted_auto_dip_slip_lo)
                / (self._scripted_auto_dip_slip_hi - self._scripted_auto_dip_slip_lo)
            )
            park = _smoothstep(
                (speed - self._scripted_auto_park_v_lo)
                / (self._scripted_auto_park_v_hi - self._scripted_auto_park_v_lo)
            )
            commit = commit * (1.0 - park * crossing)

        arrived = (
            (phase == self._PHASE_APPROACH)
            & (torch.abs(tcp_pos[:, 0] - grip_x) < self._close_x_tol)
            & (along.abs() < self._close_y_tol)
            & (rel_z.abs() < self._close_z_tol)
            & (rot_cos > self._close_rot_cos)
        )
        phase = torch.where(arrived, torch.full_like(phase, self._PHASE_CLOSE), phase)

        in_close = phase == self._PHASE_CLOSE
        self._sep_close_counter = torch.where(
            in_close, self._sep_close_counter + 1, torch.zeros_like(self._sep_close_counter)
        )
        to_lift = in_close & (self._sep_close_counter >= self._scripted_close_steps)
        phase = torch.where(to_lift, torch.full_like(phase, self._PHASE_LIFT), phase)

        to_hold = (phase == self._PHASE_LIFT) & (tcp_pos[:, 2] > deck_z + self._scripted_hold_z_above)
        phase = torch.where(to_hold, torch.full_like(phase, self._PHASE_HOLD), phase)
        self._sep_phase = phase

        advance_x = stand_x + (grip_x - stand_x) * commit
        dx = advance_x - tcp_pos[:, 0]
        guarded = tcp_pos[:, 0] > stand_x - self._lateral_guard_gap
        dx = torch.where(guarded, dx.clamp(-self._max_lateral_step, self._max_lateral_step), dx)
        target_world = torch.stack([tcp_pos[:, 0] + dx, aim_y, lip_z], dim=-1)

        is_close = phase == self._PHASE_CLOSE
        close_y = sheet_pos[:, 1] + sign * speed * self._scripted_comove_lead_time
        target_world[:, 0] = torch.where(is_close, grip_x, target_world[:, 0])
        target_world[:, 1] = torch.where(is_close, close_y, target_world[:, 1])
        target_world[:, 2] = torch.where(is_close, lip_z, target_world[:, 2])

        is_lift_or_hold = (phase == self._PHASE_LIFT) | (phase == self._PHASE_HOLD)
        target_world[:, 0] = torch.where(is_lift_or_hold, tcp_pos[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(is_lift_or_hold, tcp_pos[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(
            is_lift_or_hold,
            torch.full_like(target_world[:, 2], deck_z + self._scripted_lift_z_above),
            target_world[:, 2],
        )

        pos_input = self._position_input(target_world, tcp_pos)
        rot_input = self._rotation_input(tcp_q)

        _open_val, close_val = self._scripted_gripper_values()
        entry = self._gripper_action_for_aperture(self._entry_aperture)
        gripper = torch.where(
            phase == self._PHASE_APPROACH,
            torch.full((self.num_envs,), entry, device=device, dtype=torch.float32),
            torch.full((self.num_envs,), close_val, device=device, dtype=torch.float32),
        ).unsqueeze(-1)

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
