from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


@dataclass(frozen=True)
class _AimPoint:
    pose: Pose


class ScriptedDoorCloseMixin(_ScriptedInsertFamilyMixin):
    _PHASE_SWEEP = 0
    _PHASE_UNLOAD = 1
    _PHASE_WITHDRAW = 2

    cabinet: Actor
    door: Actor
    _pin_x: float
    _pin_y: float

    _scripted_hover_z_above: float = 0.20
    _close_wait_clearance: float = 0.03
    _close_commit_radius: float = 0.014
    _close_commit_ramp: float = 0.014

    _close_stand_start: float = 0.020
    _close_close_rate: float = 0.004
    _close_arm_xy_tol: float = 0.020
    _close_arm_z_tol: float = 0.015

    _close_arc_rate: float = 0.12
    _close_arc_lead: float = 0.16
    _close_arc_overshoot: float = 0.03
    _close_seat_angle: float = 0.10
    _close_seat_steps: int = 2
    _close_sweep_budget: int = 130
    _close_reach_budget: float = 0.80

    _close_unload_stand: float = 0.030
    _close_unload_rate: float = 0.010
    _close_unload_steps: int = 4
    _close_release_radius: float = 0.030
    _close_release_radius_rate: float = 0.010

    _close_sweep_scale_floor: float = 3.5

    _close_withdraw_z_above: float = 0.25
    _close_withdraw_track_z: float = 0.12

    _close_contact_radius: float = 0.065
    _close_contact_offset: float = 0.014
    _close_contact_z: float = 0.070

    def _door_angle(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def _scripted_rider(self) -> Actor:
        rot = quaternion_to_matrix(self.cabinet.pose.q)
        off = self._close_contact_offset + self._close_stand
        cos_p, sin_p = torch.cos(self._close_phi), torch.sin(self._close_phi)
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 0] = self._pin_x - self._close_radius * cos_p + off * sin_p
        local[:, 1] = self._pin_y + self._close_radius * sin_p + off * cos_p
        local[:, 2] = self._close_contact_z
        p = self.cabinet.pose.p + torch.bmm(rot, local.unsqueeze(-1)).squeeze(-1)
        return cast("Actor", _AimPoint(Pose.create_from_pq(p)))

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._close_commit_radius

    @property
    def _scripted_socket_chamfer(self) -> float:
        return self._close_commit_ramp

    def _scripted_wait_lower_z(
        self, deck_z: float, rider_pos: torch.Tensor
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        return rider_pos[:, 2] + self._close_wait_clearance, rider_pos[:, 2]

    def _scripted_hold_gripper(self) -> torch.Tensor:
        _, close_val = self._scripted_gripper_values()
        return torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_close_phase"):
            self._close_phase = torch.full((b,), self._PHASE_SWEEP, dtype=torch.long, device=device)
            self._close_engaged = torch.zeros((b,), dtype=torch.bool, device=device)
            self._close_phi = torch.zeros((b,), device=device)
            self._close_stand = torch.zeros((b,), device=device)
            self._close_radius = torch.zeros((b,), device=device)
            self._close_steps = torch.zeros((b,), dtype=torch.long, device=device)
            self._close_seat_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._close_unload_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._close_withdraw_xy = torch.zeros((b, 2), device=device)
        angle = self._door_angle().detach()
        if env_idx is None:
            self._close_phase[:] = self._PHASE_SWEEP
            self._close_engaged[:] = False
            self._close_phi[:] = angle
            self._close_stand[:] = self._close_stand_start
            self._close_radius[:] = self._close_contact_radius
            self._close_steps[:] = 0
            self._close_seat_counter[:] = 0
            self._close_unload_counter[:] = 0
        else:
            self._close_phase[env_idx] = self._PHASE_SWEEP
            self._close_engaged[env_idx] = False
            self._close_phi[env_idx] = angle[env_idx]
            self._close_stand[env_idx] = self._close_stand_start
            self._close_radius[env_idx] = self._close_contact_radius
            self._close_steps[env_idx] = 0
            self._close_seat_counter[env_idx] = 0
            self._close_unload_counter[env_idx] = 0

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_close_phase"):
            self.reset_scripted()

        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        state = self._intercept_state()
        door_angle = self._door_angle()

        lower_z = cast("torch.Tensor", state.lower_z)
        in_sweep = self._close_phase == self._PHASE_SWEEP
        arrived = (torch.abs(state.tcp_pos[:, 2] - lower_z) < self._close_arm_z_tol) & (
            state.xy_dist < self._close_arm_xy_tol
        )
        self._close_engaged = self._close_engaged | arrived
        armed = in_sweep & self._close_engaged
        self._close_stand = torch.where(
            armed, (self._close_stand - self._close_close_rate).clamp(min=0.0), self._close_stand
        )
        sweeping = armed & (self._close_stand <= 0.0)
        self._close_steps = torch.where(armed, self._close_steps + 1, self._close_steps)
        engaged = self._close_steps > 0
        seated = engaged & (door_angle < self._close_seat_angle)
        self._close_seat_counter = torch.where(
            seated, self._close_seat_counter + 1, torch.zeros_like(self._close_seat_counter)
        )

        reach = torch.linalg.norm(state.tcp_pos - self.agent.robot.pose.p, dim=-1)
        out_of_arm = engaged & (reach > self._close_reach_budget) & (torch.abs(self.belt.velocity) > 1e-6)

        to_unload = in_sweep & (
            (self._close_seat_counter >= self._close_seat_steps)
            | (self._close_steps > self._close_sweep_budget)
            | out_of_arm
        )
        self._close_phase = torch.where(
            to_unload, torch.full_like(self._close_phase, self._PHASE_UNLOAD), self._close_phase
        )
        in_unload = self._close_phase == self._PHASE_UNLOAD
        self._close_unload_counter = torch.where(
            in_unload, self._close_unload_counter + 1, torch.zeros_like(self._close_unload_counter)
        )
        self._close_stand = torch.where(
            in_unload,
            (self._close_stand + self._close_unload_rate).clamp(max=self._close_unload_stand),
            self._close_stand,
        )
        to_withdraw = in_unload & (self._close_unload_counter >= self._close_unload_steps)
        self._close_phase = torch.where(
            to_withdraw, torch.full_like(self._close_phase, self._PHASE_WITHDRAW), self._close_phase
        )
        is_withdraw = self._close_phase == self._PHASE_WITHDRAW
        self._close_radius = torch.where(
            in_unload | is_withdraw,
            (self._close_radius - self._close_release_radius_rate).clamp(min=self._close_release_radius),
            self._close_radius,
        )

        if self._scripted_auto_scale is not None:
            pushing = self._close_engaged & ~is_withdraw & (torch.abs(self.belt.velocity) > 1e-6)
            self._scripted_auto_scale = torch.where(
                pushing, self._scripted_auto_scale.clamp(min=self._close_sweep_scale_floor), self._scripted_auto_scale
            )

        servo = torch.clamp(
            self._close_phi - self._close_arc_rate, min=door_angle - self._close_arc_lead, max=door_angle
        )
        self._close_phi = torch.where(
            sweeping,
            servo.clamp(min=-self._close_arc_overshoot),
            door_angle,
        )

        aim = self._scripted_rider.pose.p
        low = is_withdraw & (state.tcp_pos[:, 2] < deck_z + self._close_withdraw_track_z)
        self._close_withdraw_xy = torch.where((to_withdraw | low).unsqueeze(-1), aim[:, :2], self._close_withdraw_xy)
        rise_z = torch.where(
            low,
            torch.full_like(state.tcp_pos[:, 2], deck_z + self._close_withdraw_track_z),
            torch.full_like(state.tcp_pos[:, 2], deck_z + self._close_withdraw_z_above),
        )
        withdraw_target = torch.stack([self._close_withdraw_xy[:, 0], self._close_withdraw_xy[:, 1], rise_z], dim=-1)
        target = torch.where(is_withdraw.unsqueeze(-1), withdraw_target, state.target_world)
        held = (self._close_engaged & ~is_withdraw & ~in_unload).unsqueeze(-1)
        target = torch.where(held, torch.cat([target[:, :2], aim[:, 2:3]], dim=-1), target)
        unload_lift = torch.stack(
            [aim[:, 0], aim[:, 1], torch.full_like(state.tcp_pos[:, 2], deck_z + self._close_withdraw_track_z)],
            dim=-1,
        )
        target = torch.where((in_unload & ~is_withdraw).unsqueeze(-1), unload_lift, target)

        pos_input = self._position_input(target, state.tcp_pos)
        rot_input = self._rotation_input(state.tcp_q)
        gripper = self._scripted_hold_gripper()

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
