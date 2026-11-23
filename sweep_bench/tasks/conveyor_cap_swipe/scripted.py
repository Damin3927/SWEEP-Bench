from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted import ScriptedControlMixin

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedCapSwipeMixin(ScriptedControlMixin):
    _PHASE_APPROACH = 0
    _PHASE_DESCEND = 1
    _PHASE_SWIPE = 2
    _PHASE_RETREAT = 3

    cover: Actor
    carrier: Actor
    _carrier_height: float
    _carrier_origin_above_deck: float
    _cap_swipe_push_sign: torch.Tensor

    _scripted_accel_step: float | None = 0.0025

    _scripted_approach_z_above: float = 0.12
    _scripted_standoff: float = 0.060
    _scripted_push_through: float = 0.055

    _swipe_z_above_carrier: float = 0.022

    _scripted_align_tol: float = 0.02
    _scripted_z_tol: float = 0.008

    _scripted_settle_cos: float = 0.996
    _scripted_settle_max_steps: int = 5

    _scripted_comove_lead_time: float = 0.08

    _scripted_auto_bias_gain: float = 0.3
    _scripted_auto_bias_max: float = 0.10
    _scripted_auto_bias_radius: float = 0.10
    _scripted_auto_bias_cross: float = 0.06
    _scripted_auto_bias_v_lo: float = 0.15
    _scripted_auto_bias_v_hi: float = 0.30

    _scripted_push_face_flat: bool = True

    _swipe_pos_step: float = 0.025
    _swipe_accel_step: float = 0.006

    _swipe_clear_margin: float = 0.005
    _swipe_max_steps: int = 45

    _swipe_retreat_z_above: float = 0.22

    _scripted_reach_radius: float = 0.70

    @property
    def _scripted_rider(self) -> Actor:
        return self.cover

    def _reach_limited(self, target_xy: torch.Tensor) -> torch.Tensor:
        base_xy = self.agent.robot.pose.p[:, :2]
        dx = target_xy[:, 0] - base_xy[:, 0]
        dy = target_xy[:, 1] - base_xy[:, 1]
        max_dy = torch.sqrt((self._scripted_reach_radius**2 - dx * dx).clamp(min=0.0))
        limited_y = base_xy[:, 1] + torch.clamp(dy, -max_dy, max_dy)
        riding = torch.abs(self.belt.velocity) > 0.0
        return torch.stack([target_xy[:, 0], torch.where(riding, limited_y, target_xy[:, 1])], dim=-1)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_scripted_phase"):
            self._scripted_phase = torch.full((b,), self._PHASE_APPROACH, dtype=torch.long, device=device)
            self._scripted_settle_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._scripted_swipe_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._scripted_retreat_xy = torch.zeros((b, 2), device=device)
            self._swipe_prev_delta = torch.zeros((b, 3), device=device)
        self._reset_scripted_shared(env_idx)

        if env_idx is None:
            self._scripted_phase[:] = self._PHASE_APPROACH
            self._scripted_settle_counter[:] = 0
            self._scripted_swipe_counter[:] = 0
            self._swipe_prev_delta[:] = 0.0
        else:
            self._scripted_phase[env_idx] = self._PHASE_APPROACH
            self._scripted_settle_counter[env_idx] = 0
            self._scripted_swipe_counter[env_idx] = 0
            self._swipe_prev_delta[env_idx] = 0.0

    def _swipe_position_input(
        self, target_world: torch.Tensor, tcp_pos: torch.Tensor, swiping: torch.Tensor
    ) -> torch.Tensor:
        delta = target_world - tcp_pos
        ones = torch.ones((self.num_envs,), dtype=delta.dtype, device=delta.device)
        scale = ones if self._scripted_auto_scale is None else self._scripted_auto_scale
        auto_cap = self._scripted_pos_step * scale
        auto_accel = (self._scripted_accel_step or self._scripted_pos_step) * scale
        cap = torch.stack([self._swipe_pos_step * ones, auto_cap, auto_cap], dim=-1)
        accel = torch.stack([self._swipe_accel_step * ones, auto_accel, auto_accel], dim=-1)

        prev = self._swipe_prev_delta
        delta = prev + (delta - prev).clamp(-accel, accel)
        clamped = torch.clamp(delta, -cap, cap)
        self._swipe_prev_delta = torch.where(swiping.unsqueeze(-1), clamped, self._scripted_prev_delta)
        return torch.clip(self._apply_reach_guard(clamped, tcp_pos) / 0.1, -1.0, 1.0)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_scripted_phase"):
            self.reset_scripted()

        device = self.device
        ox, _oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        push_sign = self._cap_swipe_push_sign

        d_orient = torch.zeros((self.num_envs, 2), device=device)
        if self._scripted_push_face_flat:
            d_orient[:, 0] = 1.0
        else:
            d_orient[:, 1] = self.belt.direction_sign
        self._scripted_motion_dir = d_orient

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        cover_pos = self.cover.pose.p
        cover_v_xy = self.cover.linear_velocity[:, :2]
        carrier_pos = self.carrier.pose.p
        self._update_auto_speed_scale(tcp_pos, cover_pos)

        carrier_top_z = carrier_pos[:, 2] - self._carrier_origin_above_deck + self._carrier_height
        swipe_z = carrier_top_z + self._swipe_z_above_carrier

        beside_x = cover_pos[:, 0] - push_sign * self._scripted_standoff
        comove = self.belt.velocity * self._scripted_comove_lead_time

        if self._scripted_auto_scale is not None:
            sign = self.belt.direction_sign
            along = (tcp_pos[:, 1] - cover_pos[:, 1]) * sign
            cross = torch.abs(tcp_pos[:, 0] - beside_x)
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

        beside_xy = cover_pos[:, :2].clone()
        beside_xy[:, 0] = beside_x
        beside_xy[:, 1] = cover_pos[:, 1] + comove

        phase = self._scripted_phase

        beside_ref = torch.stack([beside_x, cover_pos[:, 1]], dim=-1)
        xy_dist_beside = torch.linalg.norm(tcp_pos[:, :2] - beside_ref, dim=-1)
        z_diff = torch.abs(tcp_pos[:, 2] - swipe_z)

        target_q = self._gripper_down_target_q()
        rot_cos = torch.abs((tcp_q * target_q).sum(dim=-1))
        hover_aligned = (phase == self._PHASE_APPROACH) & (xy_dist_beside < self._scripted_align_tol)
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

        to_swipe = (
            (phase == self._PHASE_DESCEND)
            & (xy_dist_beside < self._scripted_align_tol)
            & (z_diff < self._scripted_z_tol)
        )
        phase = torch.where(to_swipe, torch.full_like(phase, self._PHASE_SWIPE), phase)

        in_swipe = phase == self._PHASE_SWIPE
        self._scripted_swipe_counter = torch.where(
            in_swipe,
            self._scripted_swipe_counter + 1,
            torch.zeros_like(self._scripted_swipe_counter),
        )
        cover_cleared = (cover_pos[:, 0] - ox) * push_sign > self._conveyor_half_width + self._swipe_clear_margin
        to_retreat = in_swipe & (
            cover_cleared | (cover_pos[:, 2] < deck_z) | (self._scripted_swipe_counter >= self._swipe_max_steps)
        )
        self._scripted_retreat_xy = torch.where(to_retreat.unsqueeze(-1), tcp_pos[:, :2], self._scripted_retreat_xy)
        phase = torch.where(to_retreat, torch.full_like(phase, self._PHASE_RETREAT), phase)

        self._scripted_phase = phase

        target_world = tcp_pos.clone()

        lead_xy = self._lookahead_xy(cover_v_xy, xy_dist_beside, beside_ref - tcp_pos[:, :2])
        approach_xy = self._reach_limited(beside_xy + lead_xy)
        is_approach = phase == self._PHASE_APPROACH
        target_world[:, 0] = torch.where(is_approach, approach_xy[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(is_approach, approach_xy[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(
            is_approach,
            torch.full_like(target_world[:, 2], deck_z + self._scripted_approach_z_above),
            target_world[:, 2],
        )

        is_descend = phase == self._PHASE_DESCEND
        target_world[:, 0] = torch.where(is_descend, beside_xy[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(is_descend, beside_xy[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(is_descend, swipe_z, target_world[:, 2])

        swipe_x = cover_pos[:, 0] + push_sign * self._scripted_push_through
        swipe_y = cover_pos[:, 1] + comove
        is_swipe = phase == self._PHASE_SWIPE
        target_world[:, 0] = torch.where(is_swipe, swipe_x, target_world[:, 0])
        target_world[:, 1] = torch.where(is_swipe, swipe_y, target_world[:, 1])
        target_world[:, 2] = torch.where(is_swipe, swipe_z, target_world[:, 2])

        is_retreat = phase == self._PHASE_RETREAT
        target_world[:, 0] = torch.where(is_retreat, self._scripted_retreat_xy[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(is_retreat, self._scripted_retreat_xy[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(
            is_retreat,
            torch.full_like(target_world[:, 2], deck_z + self._swipe_retreat_z_above),
            target_world[:, 2],
        )

        approach_pos = self._position_input(target_world, tcp_pos)
        swipe_pos = self._swipe_position_input(target_world, tcp_pos, is_swipe)
        pos_input = torch.where(is_swipe.unsqueeze(-1), swipe_pos, approach_pos)

        rot_input = self._rotation_input(tcp_q)

        _, close_val = self._scripted_gripper_values()
        gripper = torch.full((self.num_envs, 1), close_val, device=device, dtype=torch.float32)

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
