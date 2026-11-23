from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedSwingLoadMixin(_ScriptedGraspFamilyMixin):
    frame: Actor
    _rail_center_z: float
    _pendant_rest_angle: float

    def pendant_in_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def pendant_rest_rate(self) -> float:
        raise NotImplementedError

    _scripted_approach_z_above: float = 0.20

    _scripted_close_xy_tol: float = 0.020
    _scripted_close_z_tol: float = 0.012

    _scripted_lift_z_above: float = 0.29
    _scripted_hold_z_above: float = 0.26

    _sl_lift_pos_step: float = 0.018
    _sl_lift_accel_step: float = 0.0025
    _sl_damp_accel_step: float = 0.010

    _sl_damp_gain_angle: float = 0.12
    _sl_damp_offset_max: float = 0.05
    _sl_damp_rate_window: int = 1

    _scripted_grasp_endgame_scale_floor: float = 1.5

    _scripted_reach_guard_radius: ClassVar[dict[str, float]] = {
        "panda": 0.82,
        "panda_wristcam": 0.82,
    }

    _sl_grip_aperture: float = 0.022

    @property
    def _scripted_rider(self) -> Actor:
        return self.frame

    def _scripted_gripper_values(self) -> tuple[float, float]:
        open_val, _hard_shut = super()._scripted_gripper_values()
        return open_val, float(self._gripper_action_for_aperture(self._sl_grip_aperture))

    def _scripted_grasp_z(self, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2] + self._rail_center_z

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_sl_prev_delta"):
            self._sl_prev_delta = torch.zeros((b, 3), device=device)
            self._sl_hold_anchor_xy = torch.zeros((b, 2), device=device)
            self._sl_in_hold = torch.zeros((b,), dtype=torch.bool, device=device)
            self._sl_angle_hist = torch.zeros((b, self._sl_damp_rate_window + 1), device=device)
        if env_idx is None:
            self._sl_prev_delta[:] = 0.0
            self._sl_hold_anchor_xy[:] = 0.0
            self._sl_in_hold[:] = False
            self._sl_angle_hist[:] = 0.0
        else:
            self._sl_prev_delta[env_idx] = 0.0
            self._sl_hold_anchor_xy[env_idx] = 0.0
            self._sl_in_hold[env_idx] = False
            self._sl_angle_hist[env_idx] = 0.0

    def compute_scripted_action(self) -> torch.Tensor:
        action = super().compute_scripted_action()

        is_lift = self._scripted_phase == self._PHASE_LIFT
        is_hold = self._scripted_phase == self._PHASE_HOLD
        terminal = is_lift | is_hold
        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        entering = is_hold & ~self._sl_in_hold
        self._sl_hold_anchor_xy = torch.where(entering.unsqueeze(-1), tcp_pos[:, :2], self._sl_hold_anchor_xy)
        self._sl_in_hold = self._sl_in_hold | is_hold

        target_world = tcp_pos.clone()
        target_world[:, 2] = deck_z + self._scripted_lift_z_above
        damp_dy = self._damp_offset()
        hold_xy = self._sl_hold_anchor_xy.clone()
        hold_xy[:, 1] = hold_xy[:, 1] + damp_dy
        target_world[:, 0] = torch.where(is_hold, hold_xy[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(is_hold, hold_xy[:, 1], target_world[:, 1])

        accel = torch.where(
            is_hold,
            torch.full_like(damp_dy, self._sl_damp_accel_step),
            torch.full_like(damp_dy, self._sl_lift_accel_step),
        ).unsqueeze(-1)
        pos_input = self._terminal_position_input(target_world, tcp_pos, terminal, accel)
        return torch.cat([torch.where(terminal.unsqueeze(-1), pos_input, action[:, :3]), action[:, 3:]], dim=-1)

    def _damp_offset(self) -> torch.Tensor:
        angle, _pin_offset = self.pendant_in_frame()
        self._sl_angle_hist = torch.cat([self._sl_angle_hist[:, 1:], angle.unsqueeze(-1)], dim=1)
        rate = (self._sl_angle_hist[:, -1] - self._sl_angle_hist[:, 0]) * self.control_freq / self._sl_damp_rate_window

        at_rest = (torch.abs(angle) <= self._pendant_rest_angle) & (torch.abs(rate) <= self.pendant_rest_rate)
        offset = (self._sl_damp_gain_angle * angle).clamp(-self._sl_damp_offset_max, self._sl_damp_offset_max)
        return torch.where(at_rest, torch.zeros_like(offset), offset)

    def _terminal_position_input(
        self,
        target_world: torch.Tensor,
        tcp_pos: torch.Tensor,
        terminal: torch.Tensor,
        accel: torch.Tensor,
    ) -> torch.Tensor:
        prev = self._sl_prev_delta
        delta = prev + ((target_world - tcp_pos) - prev).clamp(-accel, accel)
        cap = self._sl_lift_pos_step
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap / xy_norm, max=1.0)
        z = delta[:, 2:3].clamp(-cap, cap)
        clamped = torch.cat([xy, z], dim=-1)
        if self.velocity_randomization.enabled:
            clamped = clamped * self._velrand_scale()
        self._sl_prev_delta = torch.where(terminal.unsqueeze(-1), clamped, self._scripted_prev_delta)
        return torch.clip(self._apply_reach_guard(clamped, tcp_pos) / 0.1, -1.0, 1.0)
