from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    _ScriptedInterceptMixin,
    _smoothstep,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedShroudBoreMixin(_ScriptedInterceptMixin):
    block: Actor
    rod: Actor
    _rod_radius: float
    _bore_radius: float
    _bore_length: float
    _lead_depth: float
    _mouth_radius: float
    _seat_depth: float
    _port_offset_x: float

    _scripted_hold_margin: float = 0.0035

    _scripted_pos_step: float = 0.015
    _scripted_v_ee_max: float = 0.225
    _scripted_accel_step: float | None = 0.00375
    _scripted_v_align: float = 0.15
    _scripted_max_lead: float = 0.30
    _scripted_cross_scale: float = 0.05
    _scripted_prox_scale: float = 0.05

    _scripted_hover_z_above: float = 0.26

    _standoff_gap: float = 0.010

    _descend_r_in: float = 0.020
    _descend_r_out: float = 0.060

    _commit_axis_in: float = 0.003
    _commit_axis_out: float = 0.008
    _commit_y_in: float = 0.004
    _commit_y_out: float = 0.016

    _engage_scale: float = 0.010

    _commit_rot_lo: float = 0.990
    _commit_rot_hi: float = 0.996
    _wrist_hold_backoff: float = 0.012

    _max_plunge_step: float = 0.008
    _plunge_guard_gap: float = 0.004

    @property
    def _port_tilt_rad(self) -> float:
        raise NotImplementedError

    @property
    def _tip_below_tcp(self) -> float:
        raise NotImplementedError

    @property
    def nominal_origin_z(self) -> float:
        raise NotImplementedError

    def _rod_tip_world(self) -> torch.Tensor:
        raise NotImplementedError

    def _tip_axis_coords(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def _stroke(self) -> float:
        return self._standoff_gap + self._lead_depth + self._seat_depth

    def _approach_dir(self) -> torch.Tensor:
        tilt = self._port_tilt_rad
        return torch.tensor([math.sin(tilt), 0.0, -math.cos(tilt)], dtype=torch.float32, device=self.device)

    def _in_plane_normal(self) -> torch.Tensor:
        tilt = self._port_tilt_rad
        return torch.tensor([math.cos(tilt), 0.0, math.sin(tilt)], dtype=torch.float32, device=self.device)

    @property
    def _mouth_axial(self) -> float:
        return self._bore_length + self._lead_depth

    def _nominal_port_point(self, axial: float) -> torch.Tensor:
        tilt = self._port_tilt_rad
        block_p = self.block.pose.p
        point = torch.empty_like(block_p)
        point[:, 0] = block_p[:, 0] + self._port_offset_x - axial * math.sin(tilt)
        point[:, 1] = block_p[:, 1]
        point[:, 2] = self.nominal_origin_z + axial * math.cos(tilt)
        return point

    def _station_tcp(self, tip_offset: torch.Tensor) -> torch.Tensor:
        return self._nominal_port_point(self._mouth_axial + self._standoff_gap) - tip_offset

    def _gripper_down_target_q(self) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

        base_q = super()._gripper_down_target_q()
        yaw_half = -0.25 * math.pi * self.belt.direction_sign
        q_yaw = torch.zeros_like(base_q)
        q_yaw[:, 0] = torch.cos(yaw_half)
        q_yaw[:, 3] = torch.sin(yaw_half)
        pitch_half = -0.5 * self._port_tilt_rad
        q_pitch = torch.zeros_like(base_q)
        q_pitch[:, 0] = math.cos(pitch_half)
        q_pitch[:, 2] = math.sin(pitch_half)
        return quaternion_multiply(q_pitch, quaternion_multiply(q_yaw, base_q))

    def _scripted_hold_gripper(self) -> torch.Tensor:
        target = self._rod_radius - self._scripted_hold_margin
        value = self._gripper_action_for_aperture(2.0 * target)
        return torch.full((self.num_envs, 1), value, device=self.device, dtype=torch.float32)

    @property
    def _scripted_rider(self) -> Actor:
        return self.block

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tip_offset = self._rod_tip_world() - cast("Panda", self.agent).tcp.pose.p
        station_z = self._station_tcp(tip_offset)[:, 2]
        return station_z, station_z - self._stroke * math.cos(self._port_tilt_rad)

    def _scripted_commit_window(self) -> tuple[float, float]:
        return self._commit_axis_in, self._commit_axis_out

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
        self._reset_scripted_shared(env_idx)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_scripted_motion_dir"):
            self.reset_scripted()

        device = self.device
        tilt = self._port_tilt_rad
        sin_t, cos_t = math.sin(tilt), math.cos(tilt)
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        hover_z = deck_z + self._scripted_hover_z_above

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        tcp_vel = agent.tcp.linear_velocity

        sign = self.belt.direction_sign
        speed = torch.abs(self.belt.velocity)
        motion_dir = torch.zeros((self.num_envs, 2), device=device)
        motion_dir[:, 1] = sign
        self._scripted_motion_dir = motion_dir

        u = self._approach_dir()
        e1 = self._in_plane_normal()
        stroke = self._stroke

        tip_pos = self._rod_tip_world()
        tip_standoff = ((tip_pos - self._nominal_port_point(self._mouth_axial)) @ u).neg()
        station = self._station_tcp(tip_pos - tcp_pos)
        self._update_auto_speed_scale(tcp_pos, station)

        rel = tcp_pos - station
        err_axis = torch.abs(rel @ e1)
        axial = rel @ u
        along = rel[:, 1] * sign

        backoff = (hover_z - station[:, 2]).clamp(min=0.0) / cos_t

        v_align = self._auto_scaled(self._scripted_v_align)
        max_lead = self._auto_scaled(self._scripted_max_lead)
        align_gap = err_axis + (-axial).clamp(min=0.0)
        lead = speed * self._scripted_comove_lead_time + speed * align_gap / v_align
        if self._scripted_auto_scale is not None:
            near = (along.abs() < self._scripted_auto_bias_radius) & (err_axis < self._scripted_cross_scale)
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
        aim_y = station[:, 1] + sign * lead

        rel_track = tcp_pos[:, 0] - station[:, 0]
        track_advance = torch.clamp(rel_track, -backoff * sin_t, torch.full_like(rel_track, stroke * sin_t))
        cross = torch.abs(rel_track - track_advance)
        prox_gap = torch.sqrt(cross**2 + (tcp_pos[:, 1] - aim_y) ** 2)
        descend = 1.0 - _smoothstep((prox_gap - self._descend_r_in) / (self._descend_r_out - self._descend_r_in))

        _depth, tip_radial = self._tip_axis_coords()
        clear = self._bore_radius - self._rod_radius
        captured = 1.0 - _smoothstep((tip_radial - clear) / (self._mouth_radius - clear))
        engaged = _smoothstep(-tip_standoff / self._engage_scale) * captured
        s_axis = 1.0 - _smoothstep((err_axis - self._commit_axis_in) / (self._commit_axis_out - self._commit_axis_in))
        s_y = 1.0 - _smoothstep((along.abs() - self._commit_y_in) / (self._commit_y_out - self._commit_y_in))
        rot_cos = torch.abs((tcp_q * self._gripper_down_target_q()).sum(dim=-1))
        s_wrist = _smoothstep((rot_cos - self._commit_rot_lo) / (self._commit_rot_hi - self._commit_rot_lo))
        commit = torch.maximum(s_axis * s_y * s_wrist, engaged)
        park = torch.zeros_like(commit)
        if self._scripted_auto_scale is not None:
            park = _smoothstep(
                (speed - self._scripted_auto_park_v_lo)
                / (self._scripted_auto_park_v_hi - self._scripted_auto_park_v_lo)
            )
            slip = torch.abs(tcp_vel[:, 1] * sign - speed)
            crossing = _smoothstep(
                (slip - self._scripted_auto_dip_slip_lo)
                / (self._scripted_auto_dip_slip_hi - self._scripted_auto_dip_slip_lo)
            )
            commit = commit * (1.0 - park * crossing)

        s_A = torch.maximum(torch.maximum(descend, commit), park)
        axial_target = commit * stroke - backoff * (1.0 - s_A) - self._wrist_hold_backoff * (1.0 - s_wrist)
        target = station + axial_target.unsqueeze(-1) * u
        target[:, 1] = aim_y

        delta = target - tcp_pos
        axial_delta = delta @ u
        guarded = tip_standoff < self._plunge_guard_gap
        capped = torch.where(guarded, axial_delta.clamp(-self._max_plunge_step, self._max_plunge_step), axial_delta)
        target = tcp_pos + delta + (capped - axial_delta).unsqueeze(-1) * u

        pos_input = self._position_input(target, tcp_pos)
        rot_input = self._rotation_input(tcp_q)
        gripper = self._scripted_hold_gripper()
        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
