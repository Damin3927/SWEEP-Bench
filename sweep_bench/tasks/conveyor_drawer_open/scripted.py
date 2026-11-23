from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedDrawerOpenMixin(_ScriptedInsertFamilyMixin):
    _PHASE_MERGE = 0
    _PHASE_DRAG = 1
    _PHASE_WITHDRAW = 2

    sleeve: Actor
    tray: Actor
    _engage_z_above: float
    _wait_z_above: float
    _tray_lip_top: float

    def tray_stroke(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_accel_step: float | None = 0.0025

    _scripted_hover_z_above: float = 0.20

    _drag_engage_xy_tol: float = 0.012
    _drag_engage_z_tol: float = 0.010

    _drag_rate: float = 0.006
    _drag_stroke_target: float = 0.035
    _drag_lead_margin: float = 0.010
    _drag_max_steps: int = 16

    _drawer_withdraw_z_above: float = 0.22
    _drawer_withdraw_lip_margin: float = 0.02

    @property
    def _scripted_rider(self) -> Actor:
        return self.sleeve

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._drag_engage_xy_tol - self._scripted_center_margin

    @property
    def _scripted_socket_chamfer(self) -> float:
        return 0.010

    def _tcp_lane_offset(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.sleeve.pose.q)
        agent = cast("Panda", self.agent)
        rel = agent.tcp.pose.p - self.sleeve.pose.p
        return torch.abs(torch.einsum("bj,bj->b", rot[:, :, 0], rel))

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, torch.Tensor]:
        wait_z = deck_z + self._wait_z_above
        behind_lip = self._tcp_lane_offset() < self._drag_engage_xy_tol
        return wait_z, torch.where(
            behind_lip,
            torch.full_like(behind_lip, deck_z + self._engage_z_above, dtype=torch.float32),
            torch.full_like(behind_lip, wait_z, dtype=torch.float32),
        )

    def _scripted_hold_gripper(self) -> torch.Tensor:
        _, close_val = self._scripted_gripper_values()
        return torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)

    def _sleeve_pull_dir_xy(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.sleeve.pose.q)
        pull = -rot[:, :2, 0]
        return pull / torch.linalg.norm(pull, dim=-1, keepdim=True).clamp(min=1e-9)

    def _drag_target_xy(self, state, sleeve_pos: torch.Tensor, ramp: torch.Tensor) -> torch.Tensor:
        d = state.direction
        u = self._sleeve_pull_dir_xy()
        lead = state.speed * self._scripted_comove_lead_time + self._scripted_auto_lead_bias
        lead = torch.minimum(lead, state.speed / self.control_freq + self._drag_lead_margin)
        return sleeve_pos[:, :2] + d * lead.unsqueeze(-1) + u * ramp.unsqueeze(-1)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_drawer_phase"):
            self._drawer_phase = torch.full((b,), self._PHASE_MERGE, dtype=torch.long, device=device)
            self._drag_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._drag_ramp = torch.zeros((b,), device=device)
            self._drawer_withdraw_xy = torch.zeros((b, 2), device=device)
        if env_idx is None:
            self._drawer_phase[:] = self._PHASE_MERGE
            self._drag_counter[:] = 0
            self._drag_ramp[:] = 0.0
        else:
            self._drawer_phase[env_idx] = self._PHASE_MERGE
            self._drag_counter[env_idx] = 0
            self._drag_ramp[env_idx] = 0.0

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_drawer_phase"):
            self.reset_scripted()

        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        state = self._intercept_state()
        sleeve_pos = self.sleeve.pose.p
        stroke = self.tray_stroke()

        phase = self._drawer_phase

        engaged = (
            (phase == self._PHASE_MERGE)
            & (torch.abs(state.tcp_pos[:, 2] - (deck_z + self._engage_z_above)) < self._drag_engage_z_tol)
            & (state.xy_dist < self._drag_engage_xy_tol)
            & (self._tcp_lane_offset() < self._drag_engage_xy_tol)
        )
        phase = torch.where(engaged, torch.full_like(phase, self._PHASE_DRAG), phase)

        in_drag = phase == self._PHASE_DRAG
        self._drag_counter = torch.where(in_drag, self._drag_counter + 1, torch.zeros_like(self._drag_counter))
        self._drag_ramp = torch.where(in_drag, self._drag_ramp + self._drag_rate, self._drag_ramp)
        to_withdraw = in_drag & ((stroke >= self._drag_stroke_target) | (self._drag_counter >= self._drag_max_steps))
        self._drawer_withdraw_xy = torch.where(
            to_withdraw.unsqueeze(-1), state.tcp_pos[:, :2], self._drawer_withdraw_xy
        )
        phase = torch.where(to_withdraw, torch.full_like(phase, self._PHASE_WITHDRAW), phase)
        self._drawer_phase = phase

        target = state.target_world.clone()

        drag_xy = self._drag_target_xy(state, sleeve_pos, self._drag_ramp)
        is_drag = phase == self._PHASE_DRAG
        target[:, 0] = torch.where(is_drag, drag_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_drag, drag_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_drag, torch.full_like(target[:, 2], deck_z + self._engage_z_above), target[:, 2])

        is_withdraw = phase == self._PHASE_WITHDRAW
        lip_clear = deck_z + self._tray_lip_top + self._drawer_withdraw_lip_margin
        tracking = is_withdraw & (state.tcp_pos[:, 2] < lip_clear)
        withdraw_xy = torch.where(tracking.unsqueeze(-1), drag_xy, self._drawer_withdraw_xy)
        target[:, 0] = torch.where(is_withdraw, withdraw_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_withdraw, withdraw_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(
            is_withdraw,
            torch.full_like(target[:, 2], deck_z + self._drawer_withdraw_z_above),
            target[:, 2],
        )

        pos_input = self._position_input(target, state.tcp_pos)
        rot_input = self._rotation_input(state.tcp_q)
        gripper = self._scripted_hold_gripper()

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
