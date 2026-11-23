from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedSelectorSetMixin(_ScriptedInsertFamilyMixin):
    _PHASE_APPROACH = 0
    _PHASE_DESCEND = 1
    _PHASE_SHOVE = 2
    _PHASE_WITHDRAW = 3

    carrier: Actor
    slider: Actor

    def _selector_groove_axis(self) -> torch.Tensor:
        raise NotImplementedError

    def _slider_carrier_offset(self) -> torch.Tensor:
        raise NotImplementedError

    def _selector_target_local_y(self) -> torch.Tensor:
        raise NotImplementedError

    def _selector_push_sign(self) -> torch.Tensor:
        raise NotImplementedError

    def _carrier_top_world_z(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_accel_step: float | None = 0.0025

    _selector_approach_z_above: float = 0.15
    _selector_align_standoff: float = 0.048
    _selector_contact_gap: float = 0.035
    _selector_stopped_contact_gap: float = 0.018
    _selector_shove_overshoot: float = 0.002
    _selector_shove_z_above_carrier: float = 0.024

    _selector_align_s_tol: float = 0.014
    _selector_align_x_tol: float = 0.016
    _selector_align_z_tol: float = 0.010
    _selector_align_v_tol: float = 0.06
    _selector_settle_cos: float = 0.95
    _selector_settle_max_steps: int = 8

    _selector_comove_lead_time: float = 0.08
    _selector_bias_gain: float = 0.3
    _selector_bias_max: float = 0.12
    _selector_bias_radius: float = 0.10
    _selector_bias_cross: float = 0.06

    _selector_park_y: float = -0.28
    _selector_free_y: float = -0.85
    _selector_track_y_max: float = 0.55

    _selector_shove_scale_floor: float = 3.0

    _selector_shove_stop: float = 0.007
    _selector_shove_max_steps: int = 45
    _selector_shove_max_travel: float = 0.035

    _selector_retreat_back: float = 0.14
    _selector_retreat_z_above: float = 0.22

    @property
    def _scripted_rider(self) -> Actor:
        return self.carrier

    @property
    def _scripted_socket_half_width(self) -> float:
        return 0.010

    @property
    def _scripted_socket_chamfer(self) -> float:
        return 0.0

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, torch.Tensor]:
        return deck_z + self._scripted_hover_z_above, self._selector_shove_z()

    def _scripted_hold_gripper(self) -> torch.Tensor:
        _, close_val = self._scripted_gripper_values()
        return torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)

    def _selector_shove_z(self) -> torch.Tensor:
        return self._carrier_top_world_z() + self._selector_shove_z_above_carrier

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_selector_phase"):
            self._selector_phase = torch.full((b,), self._PHASE_APPROACH, dtype=torch.long, device=device)
            self._selector_settle_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._selector_shove_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._selector_entry_u = torch.zeros((b,), device=device)
            self._selector_entry_err = torch.zeros((b,), device=device)
            self._selector_retreat_xy = torch.zeros((b, 2), device=device)
        if env_idx is None:
            self._selector_phase[:] = self._PHASE_APPROACH
            self._selector_settle_counter[:] = 0
            self._selector_shove_counter[:] = 0
        else:
            self._selector_phase[env_idx] = self._PHASE_APPROACH
            self._selector_settle_counter[env_idx] = 0
            self._selector_shove_counter[env_idx] = 0

    def _selector_track_y(self, oy: float, sign: torch.Tensor, aim_y: torch.Tensor, park: torch.Tensor) -> torch.Tensor:
        stopped = self._scripted_belt_stopped()
        speed = torch.abs(self.belt.velocity)
        bias = torch.where(stopped, 0.0, self._scripted_auto_lead_bias)
        lead = speed * self._selector_comove_lead_time + bias
        rel = (aim_y - oy) * sign + lead
        low = torch.where(park & ~stopped, self._selector_park_y, self._selector_free_y)
        return oy + torch.clamp(torch.minimum(rel, torch.full_like(rel, self._selector_track_y_max)), min=low) * sign

    def _selector_update_lead_bias(self, along: torch.Tensor, cross: torch.Tensor) -> None:
        if self._scripted_auto_scale is None:
            return
        near = (along.abs() < self._selector_bias_radius) & (cross < self._selector_bias_cross)
        self._scripted_auto_lead_bias = torch.where(
            near,
            (self._scripted_auto_lead_bias - self._selector_bias_gain * along).clamp(0.0, self._selector_bias_max),
            self._scripted_auto_lead_bias,
        )

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_selector_phase"):
            self.reset_scripted()

        _ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        carrier_pos = self.carrier.pose.p
        self._update_auto_speed_scale(tcp_pos, carrier_pos)

        groove = self._selector_groove_axis()
        local = self._slider_carrier_offset()
        s_slider = local[:, 1]
        s_target = self._selector_target_local_y()
        d = self._selector_push_sign()
        err = s_target - s_slider
        shove_z = self._selector_shove_z()

        flip = torch.where(groove[:, 1] >= 0.0, 1.0, -1.0).unsqueeze(-1)
        self._scripted_motion_dir = groove[:, :2] * flip

        rel_tcp = tcp_pos - carrier_pos
        u_tcp = (rel_tcp * groove).sum(dim=-1)
        cross_axis = torch.stack([groove[:, 1], -groove[:, 0], torch.zeros_like(groove[:, 0])], dim=-1)
        x_local = (rel_tcp * cross_axis).sum(dim=-1)

        phase = self._selector_phase

        u_stand = s_slider - d * self._selector_align_standoff
        u_contact = torch.where(
            self._scripted_belt_stopped(),
            s_target - d * (self._selector_stopped_contact_gap - self._selector_shove_overshoot),
            s_target - d * (self._selector_contact_gap - self._selector_shove_overshoot),
        )
        in_shove = phase == self._PHASE_SHOVE
        u_des = torch.where(in_shove, u_contact, u_stand)

        aim = carrier_pos + groove * u_des.unsqueeze(-1)
        sign = self.belt.direction_sign
        self._selector_update_lead_bias((tcp_pos[:, 1] - aim[:, 1]) * sign, x_local.abs())
        track_y = self._selector_track_y(oy, sign, aim[:, 1], phase == self._PHASE_APPROACH)

        target_q = self._gripper_down_target_q()
        rot_cos = torch.abs((tcp_q * target_q).sum(dim=-1))
        on_standoff = ((u_tcp - u_stand).abs() < self._selector_align_s_tol) & (
            x_local.abs() < self._selector_align_x_tol
        )
        hover_aligned = (phase == self._PHASE_APPROACH) & on_standoff
        self._selector_settle_counter = torch.where(
            hover_aligned, self._selector_settle_counter + 1, torch.zeros_like(self._selector_settle_counter)
        )
        comoving = (agent.tcp.linear_velocity[:, 1] - self.belt.velocity) * sign > -self._selector_align_v_tol
        ready = (rot_cos > self._selector_settle_cos) & comoving
        settled = ready | (self._selector_settle_counter >= self._selector_settle_max_steps)
        to_descend = hover_aligned & settled
        phase = torch.where(to_descend, torch.full_like(phase, self._PHASE_DESCEND), phase)

        to_shove = (
            (phase == self._PHASE_DESCEND)
            & on_standoff
            & (torch.abs(tcp_pos[:, 2] - shove_z) < self._selector_align_z_tol)
        )
        self._selector_entry_u = torch.where(to_shove, u_tcp, self._selector_entry_u)
        self._selector_entry_err = torch.where(to_shove, err.abs(), self._selector_entry_err)
        phase = torch.where(to_shove, torch.full_like(phase, self._PHASE_SHOVE), phase)

        in_shove = phase == self._PHASE_SHOVE
        self._selector_shove_counter = torch.where(
            in_shove, self._selector_shove_counter + 1, torch.zeros_like(self._selector_shove_counter)
        )
        advance = (u_tcp - self._selector_entry_u) * d
        to_withdraw = in_shove & (
            (err.abs() <= self._selector_shove_stop)
            | (self._selector_shove_counter >= self._selector_shove_max_steps)
            | (advance > self._selector_entry_err + self._selector_shove_max_travel)
        )
        self._selector_retreat_xy = torch.where(
            to_withdraw.unsqueeze(-1),
            tcp_pos[:, :2] - self._selector_retreat_back * d.unsqueeze(-1) * groove[:, :2],
            self._selector_retreat_xy,
        )
        phase = torch.where(to_withdraw, torch.full_like(phase, self._PHASE_WITHDRAW), phase)

        self._selector_phase = phase

        if self._scripted_auto_scale is not None:
            self._scripted_auto_scale = torch.where(
                phase == self._PHASE_SHOVE,
                self._scripted_auto_scale.clamp(min=self._selector_shove_scale_floor),
                self._scripted_auto_scale,
            )

        target = aim.clone()
        target[:, 1] = track_y

        is_approach = phase == self._PHASE_APPROACH
        target[:, 2] = torch.where(
            is_approach,
            torch.full_like(target[:, 2], deck_z + self._selector_approach_z_above),
            shove_z,
        )

        is_withdraw = phase == self._PHASE_WITHDRAW
        target[:, 0] = torch.where(is_withdraw, self._selector_retreat_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_withdraw, self._selector_retreat_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(
            is_withdraw,
            torch.full_like(target[:, 2], deck_z + self._selector_retreat_z_above),
            target[:, 2],
        )

        pos_input = self._position_input(target, tcp_pos)
        rot_input = self._rotation_input(tcp_q)
        gripper = self._scripted_hold_gripper()

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
