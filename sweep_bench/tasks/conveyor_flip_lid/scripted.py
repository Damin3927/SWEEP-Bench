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


class ScriptedFlipLidMixin(_ScriptedInsertFamilyMixin):
    _PHASE_SWEEP = 0
    _PHASE_WITHDRAW = 1

    box: Actor
    lid: Actor
    _pin_y: float
    _pin_z: float
    _contact_radius_y: float
    _contact_radius_z: float

    _scripted_hover_z_above: float = 0.20
    _flip_wait_clearance: float = 0.03
    _flip_commit_radius: float = 0.012
    _flip_commit_ramp: float = 0.012

    _flip_stand_start: float = 0.030
    _flip_close_rate: float = 0.004
    _flip_arm_xy_tol: float = 0.025
    _flip_arm_z_tol: float = 0.012

    _flip_arc_rate: float = 0.052
    _flip_arc_lead: float = 0.21
    _flip_arc_max: float = 1.66
    _flip_release_angle: float = 1.54
    _flip_sweep_budget: int = 130

    _flip_withdraw_z_above: float = 0.25

    def _flip_lid_angle(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def _scripted_rider(self) -> Actor:
        rot = quaternion_to_matrix(self.box.pose.q)
        reach = self._contact_radius_y + self._flip_stand
        cos_p, sin_p = torch.cos(self._flip_phi), torch.sin(self._flip_phi)
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 1] = self._pin_y + reach * cos_p - self._contact_radius_z * sin_p
        local[:, 2] = self._pin_z + reach * sin_p + self._contact_radius_z * cos_p
        p = self.box.pose.p + torch.bmm(rot, local.unsqueeze(-1)).squeeze(-1)
        return cast("Actor", _AimPoint(Pose.create_from_pq(p)))

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._flip_commit_radius

    @property
    def _scripted_socket_chamfer(self) -> float:
        return self._flip_commit_ramp

    def _scripted_wait_lower_z(
        self, deck_z: float, rider_pos: torch.Tensor
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        return rider_pos[:, 2] + self._flip_wait_clearance, rider_pos[:, 2]

    def _scripted_hold_gripper(self) -> torch.Tensor:
        _, close_val = self._scripted_gripper_values()
        return torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_flip_phase"):
            self._flip_phase = torch.full((b,), self._PHASE_SWEEP, dtype=torch.long, device=device)
            self._flip_phi = torch.zeros((b,), device=device)
            self._flip_stand = torch.zeros((b,), device=device)
            self._flip_steps = torch.zeros((b,), dtype=torch.long, device=device)
            self._flip_withdraw_xy = torch.zeros((b, 2), device=device)
        if env_idx is None:
            self._flip_phase[:] = self._PHASE_SWEEP
            self._flip_phi[:] = 0.0
            self._flip_stand[:] = self._flip_stand_start
            self._flip_steps[:] = 0
        else:
            self._flip_phase[env_idx] = self._PHASE_SWEEP
            self._flip_phi[env_idx] = 0.0
            self._flip_stand[env_idx] = self._flip_stand_start
            self._flip_steps[env_idx] = 0

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_flip_phase"):
            self.reset_scripted()

        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        state = self._intercept_state()
        lid_angle = self._flip_lid_angle()

        lower_z = cast("torch.Tensor", state.lower_z)
        armed = (
            (self._flip_phase == self._PHASE_SWEEP)
            & (torch.abs(state.tcp_pos[:, 2] - lower_z) < self._flip_arm_z_tol)
            & (state.xy_dist < self._flip_arm_xy_tol)
        )
        self._flip_stand = torch.where(
            armed, (self._flip_stand - self._flip_close_rate).clamp(min=0.0), self._flip_stand
        )
        sweeping = armed & (self._flip_stand <= 0.0)
        servo = torch.clamp(self._flip_phi + self._flip_arc_rate, min=lid_angle, max=lid_angle + self._flip_arc_lead)
        self._flip_phi = torch.where(sweeping, servo.clamp(0.0, self._flip_arc_max), self._flip_phi)
        self._flip_steps = torch.where(armed, self._flip_steps + 1, self._flip_steps)

        to_withdraw = (self._flip_phase == self._PHASE_SWEEP) & (
            (lid_angle > self._flip_release_angle) | (self._flip_steps > self._flip_sweep_budget)
        )
        self._flip_withdraw_xy = torch.where(to_withdraw.unsqueeze(-1), state.tcp_pos[:, :2], self._flip_withdraw_xy)
        self._flip_phase = torch.where(
            to_withdraw, torch.full_like(self._flip_phase, self._PHASE_WITHDRAW), self._flip_phase
        )

        withdraw_target = torch.stack(
            [
                self._flip_withdraw_xy[:, 0],
                self._flip_withdraw_xy[:, 1],
                torch.full_like(state.tcp_pos[:, 2], deck_z + self._flip_withdraw_z_above),
            ],
            dim=-1,
        )
        is_withdraw = (self._flip_phase == self._PHASE_WITHDRAW).unsqueeze(-1)
        target = torch.where(is_withdraw, withdraw_target, state.target_world)

        pos_input = self._position_input(target, state.tcp_pos)
        rot_input = self._rotation_input(state.tcp_q)
        gripper = self._scripted_hold_gripper()

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
