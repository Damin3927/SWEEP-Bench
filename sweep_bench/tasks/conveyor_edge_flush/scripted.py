from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted import ScriptedControlMixin

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedEdgeFlushMixin(ScriptedControlMixin):
    _PHASE_APPROACH = 0
    _PHASE_DESCEND = 1
    _PHASE_PUSH = 2
    _PHASE_DWELL = 3
    _PHASE_LIFT = 4
    _PHASE_RETREAT = 5

    tray: Actor
    slab: Actor
    _tray_half_x: float
    _rail_inner_x: float
    _slab_half_across: torch.Tensor
    _slab_half_along: torch.Tensor

    _scripted_accel_step: float | None = 0.0025

    _scripted_approach_z_above: float = 0.10
    _flush_push_z_above_floor: float = 0.020
    _flush_finger_half: float = 0.012
    _flush_descend_lane_margin: float = 0.003
    _flush_push_through: float = 0.014
    _flush_pos_step: float = 0.020
    _flush_accel_step: float = 0.005
    _flush_stop_gap: float = 0.0015
    _flush_push_max_steps: int = 55
    _flush_dwell_steps: int = 3
    _flush_lift_clear_z: float = 0.08
    _flush_retreat_z_above: float = 0.20

    _scripted_align_tol: float = 0.02
    _scripted_z_tol: float = 0.02

    _scripted_settle_cos: float = 0.996
    _scripted_settle_max_steps: int = 8

    _scripted_comove_lead_time: float = 0.08

    _scripted_auto_bias_gain: float = 0.3
    _scripted_auto_bias_max: float = 0.10
    _scripted_auto_bias_radius: float = 0.10
    _scripted_auto_bias_cross: float = 0.06
    _scripted_auto_bias_v_lo: float = 0.15
    _scripted_auto_bias_v_hi: float = 0.30

    @property
    def _flush_rail_sign(self) -> torch.Tensor:
        q = self.tray.pose.q
        r00 = 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)
        return torch.where(r00 >= 0.0, torch.ones_like(r00), -torch.ones_like(r00))

    def _flush_world_x(self, local_x: torch.Tensor | float) -> torch.Tensor:
        return self.tray.pose.p[:, 0] + self._flush_rail_sign * local_x

    def _flush_slab_local_x(self) -> torch.Tensor:
        return self._flush_rail_sign * (self.slab.pose.p[:, 0] - self.tray.pose.p[:, 0])

    def _slab_datum_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _flush_slab_half_across_at_skew(self, skew: torch.Tensor) -> torch.Tensor:
        return self._slab_half_across * skew.cos().abs() + self._slab_half_along * skew.sin().abs()

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_scripted_phase"):
            self._scripted_phase = torch.full((b,), self._PHASE_APPROACH, dtype=torch.long, device=device)
            self._scripted_settle_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._flush_push_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._flush_dwell_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._flush_prev_dx = torch.zeros((b,), device=device)
            self._flush_dwell_x = torch.zeros((b,), device=device)
            self._flush_hold_xy = torch.zeros((b, 2), device=device)
            self._flush_clear_x = torch.zeros((b,), device=device)
        self._reset_scripted_shared(env_idx)

        if env_idx is None:
            self._scripted_phase[:] = self._PHASE_APPROACH
            self._scripted_settle_counter[:] = 0
            self._flush_push_counter[:] = 0
            self._flush_dwell_counter[:] = 0
            self._flush_prev_dx[:] = 0.0
        else:
            self._scripted_phase[env_idx] = self._PHASE_APPROACH
            self._scripted_settle_counter[env_idx] = 0
            self._flush_push_counter[env_idx] = 0
            self._flush_dwell_counter[env_idx] = 0
            self._flush_prev_dx[env_idx] = 0.0

    def _flush_position_input(self, target_world: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
        delta = target_world - tcp_pos
        accel = self._flush_accel_step
        dx = self._flush_prev_dx + (delta[:, 0] - self._flush_prev_dx).clamp(-accel, accel)
        dx = dx.clamp(-self._flush_pos_step, self._flush_pos_step)
        self._flush_prev_dx = dx
        cap = torch.as_tensor(
            self._auto_scaled(self._scripted_pos_step), dtype=delta.dtype, device=delta.device
        ).reshape(-1, 1)
        yz = delta[:, 1:].clamp(-cap, cap)
        clamped = torch.cat([dx.unsqueeze(-1), yz], dim=-1)
        return torch.clip(clamped / 0.1, -1.0, 1.0)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_scripted_phase"):
            self.reset_scripted()

        device = self.device

        d_orient = torch.zeros((self.num_envs, 2), device=device)
        d_orient[:, 1] = self.belt.direction_sign
        self._scripted_motion_dir = d_orient

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        slab_pos = self.slab.pose.p
        slab_v_xy = self.slab.linear_velocity[:, :2]
        self._update_auto_speed_scale(tcp_pos, slab_pos)

        _ox, _oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        push_z = self.tray.pose.p[:, 2] + self._flush_push_z_above_floor

        lane_local_x = -(self._tray_half_x + self._flush_finger_half + self._flush_descend_lane_margin)
        lane_x = self._flush_world_x(lane_local_x)

        comove = self.belt.velocity * self._scripted_comove_lead_time

        if self._scripted_auto_scale is not None:
            sign = self.belt.direction_sign
            along = (tcp_pos[:, 1] - slab_pos[:, 1]) * sign
            cross = torch.abs(tcp_pos[:, 0] - lane_x)
            near = (along.abs() < self._scripted_auto_bias_radius) & (cross < self._scripted_auto_bias_cross)
            bias = torch.where(
                near,
                (self._scripted_auto_lead_bias - self._scripted_auto_bias_gain * along).clamp(
                    0.0, self._scripted_auto_bias_max
                ),
                self._scripted_auto_lead_bias,
            )
            self._scripted_auto_lead_bias = bias
            ramp = (
                (torch.abs(self.belt.velocity) - self._scripted_auto_bias_v_lo)
                / (self._scripted_auto_bias_v_hi - self._scripted_auto_bias_v_lo)
            ).clamp(0.0, 1.0)
            speed_ramp = ramp * ramp * (3.0 - 2.0 * ramp)
            comove = comove + sign * bias * speed_ramp

        phase = self._scripted_phase

        lane_ref = torch.stack([lane_x, slab_pos[:, 1]], dim=-1)
        xy_dist_lane = torch.linalg.norm(tcp_pos[:, :2] - lane_ref, dim=-1)
        z_diff = torch.abs(tcp_pos[:, 2] - push_z)

        target_q = self._gripper_down_target_q()
        rot_cos = torch.abs((tcp_q * target_q).sum(dim=-1))
        hover_aligned = (phase == self._PHASE_APPROACH) & (xy_dist_lane < self._scripted_align_tol)
        self._scripted_settle_counter = torch.where(
            hover_aligned,
            self._scripted_settle_counter + 1,
            torch.zeros_like(self._scripted_settle_counter),
        )
        settled = (rot_cos > self._scripted_settle_cos) | (
            self._scripted_settle_counter >= self._scripted_settle_max_steps
        )
        to_descend = hover_aligned & settled
        phase = torch.where(to_descend, torch.full_like(phase, self._PHASE_DESCEND), phase)

        to_push = (
            (phase == self._PHASE_DESCEND) & (xy_dist_lane < self._scripted_align_tol) & (z_diff < self._scripted_z_tol)
        )
        phase = torch.where(to_push, torch.full_like(phase, self._PHASE_PUSH), phase)

        edge_gap, skew, _bottom_z = self._slab_datum_state()
        slab_local_x = self._flush_slab_local_x()
        half_across_skew = self._flush_slab_half_across_at_skew(skew)
        in_push = phase == self._PHASE_PUSH
        self._flush_push_counter = torch.where(
            in_push, self._flush_push_counter + 1, torch.zeros_like(self._flush_push_counter)
        )
        to_dwell = in_push & (
            (edge_gap < self._flush_stop_gap) | (self._flush_push_counter >= self._flush_push_max_steps)
        )
        self._flush_dwell_x = torch.where(to_dwell, tcp_pos[:, 0], self._flush_dwell_x)
        self._flush_prev_dx = torch.where(to_dwell, torch.zeros_like(self._flush_prev_dx), self._flush_prev_dx)
        phase = torch.where(to_dwell, torch.full_like(phase, self._PHASE_DWELL), phase)

        in_dwell = phase == self._PHASE_DWELL
        self._flush_dwell_counter = torch.where(
            in_dwell, self._flush_dwell_counter + 1, torch.zeros_like(self._flush_dwell_counter)
        )
        to_lift = in_dwell & (self._flush_dwell_counter >= self._flush_dwell_steps)
        self._flush_hold_xy = torch.where(to_lift.unsqueeze(-1), tcp_pos[:, :2], self._flush_hold_xy)
        self._flush_clear_x = torch.where(to_lift, lane_x, self._flush_clear_x)
        phase = torch.where(to_lift, torch.full_like(phase, self._PHASE_LIFT), phase)

        to_retreat = (phase == self._PHASE_LIFT) & (tcp_pos[:, 2] > deck_z + self._flush_lift_clear_z)
        phase = torch.where(to_retreat, torch.full_like(phase, self._PHASE_RETREAT), phase)

        self._scripted_phase = phase

        target_world = tcp_pos.clone()

        lead_xy = self._lookahead_xy(slab_v_xy, xy_dist_lane, lane_ref - tcp_pos[:, :2])
        is_approach = phase == self._PHASE_APPROACH
        target_world[:, 0] = torch.where(is_approach, lane_x + lead_xy[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(is_approach, slab_pos[:, 1] + comove + lead_xy[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(
            is_approach,
            torch.full_like(target_world[:, 2], deck_z + self._scripted_approach_z_above),
            target_world[:, 2],
        )

        is_descend = phase == self._PHASE_DESCEND
        target_world[:, 0] = torch.where(is_descend, lane_x, target_world[:, 0])
        target_world[:, 1] = torch.where(is_descend, slab_pos[:, 1] + comove, target_world[:, 1])
        target_world[:, 2] = torch.where(is_descend, push_z, target_world[:, 2])

        contact_local_x = slab_local_x - half_across_skew - self._flush_finger_half
        push_x = self._flush_world_x(contact_local_x + self._flush_push_through)
        is_push = phase == self._PHASE_PUSH
        is_dwell = phase == self._PHASE_DWELL
        is_stroke = is_push | is_dwell
        target_world[:, 0] = torch.where(is_push, push_x, target_world[:, 0])
        target_world[:, 0] = torch.where(is_dwell, self._flush_dwell_x, target_world[:, 0])
        target_world[:, 1] = torch.where(is_stroke, slab_pos[:, 1] + comove, target_world[:, 1])
        target_world[:, 2] = torch.where(is_stroke, push_z, target_world[:, 2])

        is_lift = phase == self._PHASE_LIFT
        is_retreat = phase == self._PHASE_RETREAT
        endgame = is_lift | is_retreat
        target_world[:, 0] = torch.where(is_lift, self._flush_hold_xy[:, 0], target_world[:, 0])
        target_world[:, 0] = torch.where(is_retreat, self._flush_clear_x, target_world[:, 0])
        target_world[:, 1] = torch.where(endgame, self._flush_hold_xy[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(
            endgame,
            torch.full_like(target_world[:, 2], deck_z + self._flush_retreat_z_above),
            target_world[:, 2],
        )

        pos_input = torch.where(
            is_stroke.unsqueeze(-1),
            self._flush_position_input(target_world, tcp_pos),
            self._position_input(target_world, tcp_pos),
        )
        self._flush_prev_dx = torch.where(is_stroke, self._flush_prev_dx, torch.zeros_like(self._flush_prev_dx))

        rot_input = self._rotation_input(tcp_q)

        _, close_val = self._scripted_gripper_values()
        gripper = torch.full((self.num_envs, 1), close_val, device=device, dtype=torch.float32)

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
