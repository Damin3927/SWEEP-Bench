from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply, quaternion_to_matrix

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedUprightBottleMixin(_ScriptedGraspFamilyMixin):
    bottle: Actor
    _rider_half_long: torch.Tensor
    _upright_tol_deg: float

    def rider_lowest_z(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_approach_z_above: float = 0.14
    _scripted_close_xy_tol: float = 0.018
    _scripted_lift_z_above: float = 0.18
    _scripted_hold_z_above: float = 0.14
    _scripted_pos_step = 0.014
    _scripted_v_ee_max = 0.21

    _UB_GRASPING = 0
    _UB_SECURE = 1
    _UB_CARRY = 2
    _UB_PITCH = 3
    _UB_APPROACH = 4
    _UB_SETDOWN = 5
    _UB_MERGE = 6
    _UB_RELEASE = 7
    _UB_RETREAT = 8

    _ub_transport_pos_step: float = 0.045
    _ub_transport_accel_step: float = 0.006
    _ub_comove_headroom: float = 3.0
    _ub_comove_accel_step: float = 0.015

    _ub_station_dx: float = -0.28
    _ub_station_y_mag: float = 0.35
    _ub_station_z_above: float = 0.30

    _ub_secure_settle_speed: float = 0.15
    _ub_secure_max_steps: int = 20
    _ub_grasp_check_dist: float = 0.06

    _ub_pitch_gate_y: float = 0.35
    _ub_pitch_gate_z_above: float = 0.24
    _ub_pitch_gate_dx: float = 0.14
    _ub_pitch_done_cos: float = math.cos(math.radians(30.0))

    _ub_drop_hover: float = 0.08
    _ub_over_staging_tol: float = 0.02
    _ub_slip_tol: float = 0.03
    _ub_merge_max_steps: int = 25
    _ub_setdown_press: float = 0.010
    _ub_release_gap: float = 0.012
    _ub_release_steps: int = 5
    _ub_retreat_clear_margin: float = 0.05
    _ub_retreat_z_above: float = 0.30

    @property
    def _scripted_rider(self) -> Actor:
        return self.bottle

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_ub_phase"):
            self._ub_phase = torch.full((b,), self._UB_GRASPING, dtype=torch.long, device=device)
            self._ub_secure_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._ub_merge_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._ub_release_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._ub_secure_xy = torch.zeros((b, 2), device=device)
            self._ub_side = torch.ones((b,), device=device)
            self._ub_track_y = torch.zeros((b,), device=device)
            self._ub_prev_delta = torch.zeros((b, 3), device=device)
        if env_idx is None:
            self._ub_phase[:] = self._UB_GRASPING
            self._ub_secure_counter[:] = 0
            self._ub_merge_counter[:] = 0
            self._ub_release_counter[:] = 0
            self._ub_side[:] = 1.0
            self._ub_track_y[:] = 0.0
            self._ub_prev_delta[:] = 0.0
        else:
            self._ub_phase[env_idx] = self._UB_GRASPING
            self._ub_secure_counter[env_idx] = 0
            self._ub_merge_counter[env_idx] = 0
            self._ub_release_counter[env_idx] = 0
            self._ub_side[env_idx] = 1.0
            self._ub_track_y[env_idx] = 0.0
            self._ub_prev_delta[env_idx] = 0.0

    def _upright_correction_q(self) -> torch.Tensor:
        axis = quaternion_to_matrix(self.bottle.pose.q)[:, :, 0]
        axis = axis * torch.where(axis[:, 2:3] >= 0.0, 1.0, -1.0)
        up = torch.zeros_like(axis)
        up[:, 2] = 1.0
        v = torch.linalg.cross(axis, up, dim=-1)
        sin_t = torch.linalg.norm(v, dim=-1)
        theta = torch.atan2(sin_t, axis[:, 2].clamp(-1.0, 1.0))
        n = v / sin_t.clamp(min=1e-9).unsqueeze(-1)
        half = 0.5 * theta
        return torch.cat([torch.cos(half).unsqueeze(-1), n * torch.sin(half).unsqueeze(-1)], dim=-1)

    def _gripper_down_target_q(self) -> torch.Tensor:
        base_q = super()._gripper_down_target_q()
        if not hasattr(self, "_ub_phase"):
            return base_q
        half = 0.25 * math.pi
        pitch_q = torch.zeros_like(base_q)
        pitch_q[:, 0] = math.cos(half)
        pitch_q[:, 1] = math.sin(half) * self._ub_side
        pitched = quaternion_multiply(pitch_q, base_q)
        tcp_q = cast("Panda", self.agent).tcp.pose.q
        servo = quaternion_multiply(self._upright_correction_q(), tcp_q)
        target = torch.where((self._ub_phase == self._UB_PITCH).unsqueeze(-1), pitched, base_q)
        in_servo = (self._ub_phase >= self._UB_APPROACH) & (self._ub_phase <= self._UB_RELEASE)
        target = torch.where(in_servo.unsqueeze(-1), servo, target)
        return torch.where((self._ub_phase == self._UB_RETREAT).unsqueeze(-1), tcp_q, target)

    def _transport_cap(self) -> torch.Tensor:
        comove = self._ub_comove_headroom * torch.abs(self.belt.velocity) / self.control_freq
        return comove.clamp(min=self._ub_transport_pos_step).unsqueeze(-1)

    def _transport_accel(self) -> torch.Tensor:
        comove = self._ub_phase >= self._UB_MERGE
        return torch.where(comove, self._ub_comove_accel_step, self._ub_transport_accel_step).unsqueeze(-1)

    def _transport_position_input(
        self, target_world: torch.Tensor, tcp_pos: torch.Tensor, ramping: torch.Tensor
    ) -> torch.Tensor:
        delta = target_world - tcp_pos
        accel = self._transport_accel()
        prev = self._ub_prev_delta
        delta = prev + (delta - prev).clamp(min=-accel, max=accel)
        cap = self._transport_cap()
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap / xy_norm, max=1.0)
        z = delta[:, 2:3].clamp(min=-cap, max=cap)
        clamped = torch.cat([xy, z], dim=-1)
        self._ub_prev_delta = torch.where(ramping.unsqueeze(-1), clamped, torch.zeros_like(clamped))
        return torch.clip(clamped / 0.1, -1.0, 1.0)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_ub_phase"):
            self.reset_scripted()

        grasp_action = super().compute_scripted_action()

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        bottle_pos = self.bottle.pose.p
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        sign = self.belt.direction_sign
        half_long = self._rider_half_long

        ub = self._ub_phase

        grasp_done = self._scripted_phase == self._PHASE_HOLD
        in_hand = torch.linalg.norm(tcp_pos - bottle_pos, dim=-1) < self._ub_grasp_check_dist
        to_secure = (ub == self._UB_GRASPING) & grasp_done & in_hand
        self._ub_secure_xy = torch.where(to_secure.unsqueeze(-1), tcp_pos[:, :2], self._ub_secure_xy)
        self._ub_side = torch.where(to_secure, -sign, self._ub_side)
        ub = torch.where(to_secure, torch.full_like(ub, self._UB_SECURE), ub)

        in_secure = ub == self._UB_SECURE
        self._ub_secure_counter = torch.where(
            in_secure,
            self._ub_secure_counter + 1,
            torch.zeros_like(self._ub_secure_counter),
        )
        agent_speed = torch.linalg.norm(agent.tcp.linear_velocity, dim=-1)
        settled_arm = (agent_speed < self._ub_secure_settle_speed) | (
            self._ub_secure_counter >= self._ub_secure_max_steps
        )
        ub = torch.where(in_secure & settled_arm, torch.full_like(ub, self._UB_CARRY), ub)

        carried_back = (
            ((tcp_pos[:, 1] - oy) * sign <= self._ub_pitch_gate_y)
            & (tcp_pos[:, 2] >= deck_z + self._ub_pitch_gate_z_above)
            & (tcp_pos[:, 0] <= ox - self._ub_pitch_gate_dx)
        )
        ub = torch.where((ub == self._UB_CARRY) & carried_back, torch.full_like(ub, self._UB_PITCH), ub)

        long_axis_z = quaternion_to_matrix(self.bottle.pose.q)[:, 2, 0]
        pitched_up = long_axis_z.abs() > self._ub_pitch_done_cos
        ub = torch.where((ub == self._UB_PITCH) & pitched_up, torch.full_like(ub, self._UB_APPROACH), ub)

        staging_y = oy + self._ub_station_y_mag * self._ub_side
        over_staging = (
            torch.linalg.norm(
                torch.stack([tcp_pos[:, 0] - ox, tcp_pos[:, 1] - staging_y], dim=-1),
                dim=-1,
            )
            < self._ub_over_staging_tol
        )
        ub = torch.where((ub == self._UB_APPROACH) & over_staging, torch.full_like(ub, self._UB_SETDOWN), ub)

        upright_now = long_axis_z.abs() > math.cos(math.radians(self._upright_tol_deg))
        landed = (self.rider_lowest_z() - deck_z) <= self._ub_release_gap
        to_merge = (ub == self._UB_SETDOWN) & landed & upright_now
        self._ub_track_y = torch.where(to_merge, tcp_pos[:, 1], self._ub_track_y)
        ub = torch.where(to_merge, torch.full_like(ub, self._UB_MERGE), ub)

        in_comove = ub >= self._UB_MERGE
        self._ub_track_y = self._ub_track_y + in_comove.float() * self.belt.velocity / self.control_freq

        slip = torch.abs(agent.tcp.linear_velocity[:, 1] - self.belt.velocity)
        in_merge = ub == self._UB_MERGE
        self._ub_merge_counter = torch.where(
            in_merge,
            self._ub_merge_counter + 1,
            torch.zeros_like(self._ub_merge_counter),
        )
        to_release = in_merge & ((slip <= self._ub_slip_tol) | (self._ub_merge_counter >= self._ub_merge_max_steps))
        ub = torch.where(to_release, torch.full_like(ub, self._UB_RELEASE), ub)

        in_release = ub == self._UB_RELEASE
        self._ub_release_counter = torch.where(
            in_release,
            self._ub_release_counter + 1,
            torch.zeros_like(self._ub_release_counter),
        )
        to_retreat = in_release & (self._ub_release_counter >= self._ub_release_steps)
        ub = torch.where(to_retreat, torch.full_like(ub, self._UB_RETREAT), ub)

        self._ub_phase = ub

        target = tcp_pos.clone()

        is_secure = ub == self._UB_SECURE
        secure_z = torch.full_like(target[:, 2], deck_z + self._scripted_lift_z_above)
        target[:, 0] = torch.where(is_secure, self._ub_secure_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_secure, self._ub_secure_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_secure, secure_z, target[:, 2])

        station_x = torch.full_like(target[:, 0], ox + self._ub_station_dx)
        station_z = torch.full_like(target[:, 2], deck_z + self._ub_station_z_above)
        at_station = (ub == self._UB_CARRY) | (ub == self._UB_PITCH)
        target[:, 0] = torch.where(at_station, station_x, target[:, 0])
        target[:, 1] = torch.where(at_station, staging_y, target[:, 1])
        target[:, 2] = torch.where(at_station, station_z, target[:, 2])

        rest_z = deck_z + half_long
        hover_z = rest_z + self._ub_drop_hover
        is_approach = ub == self._UB_APPROACH
        is_setdown = ub == self._UB_SETDOWN
        parked = is_approach | is_setdown
        target[:, 0] = torch.where(parked, torch.full_like(target[:, 0], ox), target[:, 0])
        target[:, 1] = torch.where(parked, staging_y, target[:, 1])
        target[:, 2] = torch.where(is_approach, hover_z, target[:, 2])
        target[:, 2] = torch.where(is_setdown, rest_z - self._ub_setdown_press, target[:, 2])

        is_merge = ub == self._UB_MERGE
        is_release = ub == self._UB_RELEASE
        tracking = is_merge | is_release
        target[:, 0] = torch.where(tracking, torch.full_like(target[:, 0], ox), target[:, 0])
        target[:, 1] = torch.where(tracking, self._ub_track_y, target[:, 1])
        target[:, 2] = torch.where(tracking, rest_z - self._ub_setdown_press, target[:, 2])

        is_retreat = ub == self._UB_RETREAT
        rider_top_z = deck_z + 2.0 * half_long
        still_over = is_retreat & (tcp_pos[:, 2] < rider_top_z + self._ub_retreat_clear_margin)
        target[:, 0] = torch.where(still_over, torch.full_like(target[:, 0], ox), target[:, 0])
        target[:, 1] = torch.where(still_over, self._ub_track_y, target[:, 1])
        target[:, 2] = torch.where(
            is_retreat, torch.full_like(target[:, 2], deck_z + self._ub_retreat_z_above), target[:, 2]
        )

        in_reorient = ub != self._UB_GRASPING
        pos_input = self._transport_position_input(target, tcp_pos, in_reorient)
        rot_input = self._rotation_input(tcp_q)
        open_val, close_val = self._scripted_gripper_values()
        is_open = is_release | is_retreat
        gripper = torch.where(
            is_open,
            torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32),
            torch.full((self.num_envs,), close_val, device=self.device, dtype=torch.float32),
        ).unsqueeze(-1)
        reorient_action = torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)

        return torch.where(in_reorient.unsqueeze(-1), reorient_action, grasp_action)
