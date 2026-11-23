from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedDrawerCloseMixin(_ScriptedInsertFamilyMixin):
    _PHASE_ALIGN = 0
    _PHASE_PUSH = 1
    _PHASE_WITHDRAW = 2

    sleeve: Actor
    tray: Actor
    _tray_body_half_y: float

    def _tray_stroke_axis(self) -> torch.Tensor:
        raise NotImplementedError

    def _tray_push_point(self) -> torch.Tensor:
        raise NotImplementedError

    def _tray_closure(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_accel_step: float | None = 0.0025

    _drawer_align_standoff: float = 0.045
    _drawer_align_stroke_tol: float = 0.015
    _drawer_align_y_tol: float = 0.020
    _drawer_align_z_tol: float = 0.012
    _drawer_align_v_tol: float = 0.06
    _drawer_settle_cos: float = 0.95
    _drawer_settle_max_steps: int = 20
    _drawer_comove_lead_time: float = 0.08
    _drawer_bias_gain: float = 0.3
    _drawer_bias_max: float = 0.10
    _drawer_bias_radius: float = 0.10
    _drawer_bias_cross: float = 0.06
    _drawer_park_y: float = -0.275
    _drawer_track_y_max: float = 0.55

    _drawer_push_pos_step: float = 0.018
    _drawer_push_accel_step: float = 0.005
    _drawer_push_overshoot: float = 0.004
    _drawer_push_stop: float = 0.004
    _drawer_push_max_steps: int = 45
    _drawer_push_max_travel: float = 0.040

    _drawer_retreat_back: float = 0.15
    _drawer_retreat_z_above: float = 0.22

    @property
    def _scripted_rider(self) -> Actor:
        return self.sleeve

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._tray_body_half_y

    @property
    def _scripted_socket_chamfer(self) -> float:
        return 0.0

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, torch.Tensor]:
        return deck_z + self._scripted_hover_z_above, self._tray_push_point()[:, 2]

    def _scripted_hold_gripper(self) -> torch.Tensor:
        _, close_val = self._scripted_gripper_values()
        return torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_drawer_phase"):
            self._drawer_phase = torch.full((b,), self._PHASE_ALIGN, dtype=torch.long, device=device)
            self._drawer_settle_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._drawer_push_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._drawer_push_entry_s = torch.zeros((b,), device=device)
            self._drawer_push_entry_closure = torch.zeros((b,), device=device)
            self._drawer_retreat_xy = torch.zeros((b, 2), device=device)
            self._drawer_push_prev_ds = torch.zeros((b, 1), device=device)
        if env_idx is None:
            self._drawer_phase[:] = self._PHASE_ALIGN
            self._drawer_settle_counter[:] = 0
            self._drawer_push_counter[:] = 0
            self._drawer_push_prev_ds[:] = 0.0
        else:
            self._drawer_phase[env_idx] = self._PHASE_ALIGN
            self._drawer_settle_counter[env_idx] = 0
            self._drawer_push_counter[env_idx] = 0
            self._drawer_push_prev_ds[env_idx] = 0.0

    def _drawer_track_y(self, oy: float, sign: torch.Tensor, face_y: torch.Tensor) -> torch.Tensor:
        speed = torch.abs(self.belt.velocity)
        lead = speed * self._drawer_comove_lead_time + self._scripted_auto_lead_bias
        rel = (face_y - oy) * sign + lead
        return oy + rel.clamp(self._drawer_park_y, self._drawer_track_y_max) * sign

    def _drawer_update_lead_bias(self, along: torch.Tensor, cross: torch.Tensor) -> None:
        if self._scripted_auto_scale is None:
            return
        near = (along.abs() < self._drawer_bias_radius) & (cross < self._drawer_bias_cross)
        self._scripted_auto_lead_bias = torch.where(
            near,
            (self._scripted_auto_lead_bias - self._drawer_bias_gain * along).clamp(0.0, self._drawer_bias_max),
            self._scripted_auto_lead_bias,
        )

    def _drawer_push_position_input(
        self, target_world: torch.Tensor, tcp_pos: torch.Tensor, stroke: torch.Tensor, pushing: torch.Tensor
    ) -> torch.Tensor:
        delta = target_world - tcp_pos
        s = (delta * stroke).sum(dim=-1, keepdim=True)
        rest = delta - s * stroke

        accel = self._drawer_push_accel_step
        prev = self._drawer_push_prev_ds
        s = prev + (s - prev).clamp(-accel, accel)
        s = s.clamp(-self._drawer_push_pos_step, self._drawer_push_pos_step)
        self._drawer_push_prev_ds = torch.where(pushing.unsqueeze(-1), s, torch.zeros_like(s))

        cap = self._auto_scaled(self._scripted_pos_step)
        cap_t = torch.as_tensor(
            cap.unsqueeze(-1) if isinstance(cap, torch.Tensor) else cap, dtype=delta.dtype, device=delta.device
        )
        rest = torch.clamp(rest, -cap_t, cap_t)
        return torch.clip((s * stroke + rest) / 0.1, -1.0, 1.0)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_drawer_phase"):
            self.reset_scripted()

        device = self.device
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        sleeve_pos = self.sleeve.pose.p
        self._update_auto_speed_scale(tcp_pos, sleeve_pos)

        sign = self.belt.direction_sign
        motion_dir = torch.zeros((self.num_envs, 2), device=device)
        motion_dir[:, 1] = sign
        self._scripted_motion_dir = motion_dir

        face = self._tray_push_point()
        stroke = self._tray_stroke_axis()
        closure = self._tray_closure()

        tcp_s = (tcp_pos * stroke).sum(dim=-1)
        face_s = (face * stroke).sum(dim=-1)
        stroke_gap = face_s - tcp_s
        carrier_s = (sleeve_pos * stroke).sum(dim=-1)
        tcp_s_rel = tcp_s - carrier_s

        along = (tcp_pos[:, 1] - face[:, 1]) * sign
        self._drawer_update_lead_bias(along, (stroke_gap - self._drawer_align_standoff).abs())
        track_y = self._drawer_track_y(oy, sign, face[:, 1])

        phase = self._drawer_phase

        target_q = self._gripper_down_target_q()
        rot_cos = torch.abs((tcp_q * target_q).sum(dim=-1))
        aligned = (
            (phase == self._PHASE_ALIGN)
            & (stroke_gap < self._drawer_align_standoff + self._drawer_align_stroke_tol)
            & (along.abs() < self._drawer_align_y_tol)
            & ((tcp_pos[:, 2] - face[:, 2]).abs() < self._drawer_align_z_tol)
        )
        self._drawer_settle_counter = torch.where(
            aligned, self._drawer_settle_counter + 1, torch.zeros_like(self._drawer_settle_counter)
        )
        comoving = torch.abs(agent.tcp.linear_velocity[:, 1] - self.belt.velocity) < self._drawer_align_v_tol
        ready = (rot_cos > self._drawer_settle_cos) & comoving
        settled = ready | (self._drawer_settle_counter >= self._drawer_settle_max_steps)
        to_push = aligned & settled
        self._drawer_push_entry_s = torch.where(to_push, tcp_s_rel, self._drawer_push_entry_s)
        self._drawer_push_entry_closure = torch.where(to_push, closure, self._drawer_push_entry_closure)
        phase = torch.where(to_push, torch.full_like(phase, self._PHASE_PUSH), phase)

        in_push = phase == self._PHASE_PUSH
        self._drawer_push_counter = torch.where(
            in_push, self._drawer_push_counter + 1, torch.zeros_like(self._drawer_push_counter)
        )
        advance = tcp_s_rel - self._drawer_push_entry_s
        to_withdraw = in_push & (
            (closure <= self._drawer_push_stop)
            | (self._drawer_push_counter >= self._drawer_push_max_steps)
            | (advance > self._drawer_push_entry_closure + self._drawer_push_max_travel)
        )
        self._drawer_retreat_xy = torch.where(
            to_withdraw.unsqueeze(-1),
            tcp_pos[:, :2] - self._drawer_retreat_back * stroke[:, :2],
            self._drawer_retreat_xy,
        )
        phase = torch.where(to_withdraw, torch.full_like(phase, self._PHASE_WITHDRAW), phase)

        self._drawer_phase = phase

        target = tcp_pos.clone()

        is_align = phase == self._PHASE_ALIGN
        align_target = face - self._drawer_align_standoff * stroke
        target = torch.where(is_align.unsqueeze(-1), align_target, target)
        target[:, 1] = torch.where(is_align, track_y, target[:, 1])
        target[:, 2] = torch.where(is_align, face[:, 2], target[:, 2])

        is_push = phase == self._PHASE_PUSH
        push_target = tcp_pos + (closure + self._drawer_push_overshoot).unsqueeze(-1) * stroke
        target = torch.where(is_push.unsqueeze(-1), push_target, target)
        target[:, 1] = torch.where(is_push, track_y, target[:, 1])
        target[:, 2] = torch.where(is_push, face[:, 2], target[:, 2])

        is_withdraw = phase == self._PHASE_WITHDRAW
        target[:, 0] = torch.where(is_withdraw, self._drawer_retreat_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_withdraw, self._drawer_retreat_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(
            is_withdraw,
            torch.full_like(target[:, 2], deck_z + self._drawer_retreat_z_above),
            target[:, 2],
        )

        approach_pos = self._position_input(target, tcp_pos)
        push_pos = self._drawer_push_position_input(target, tcp_pos, stroke, is_push)
        pos_input = torch.where(is_push.unsqueeze(-1), push_pos, approach_pos)

        rot_input = self._rotation_input(tcp_q)
        gripper = self._scripted_hold_gripper()

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
