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


class ScriptedToppleMixin(_ScriptedInsertFamilyMixin):
    _PHASE_PARK = 0
    _PHASE_PUSH = 1
    _PHASE_RETREAT = 2

    block: Actor
    _block_half_y: float
    _block_height: float
    _block_contact_frac: torch.Tensor

    _scripted_accel_step: float | None = 0.0025

    _scripted_contact_frac: float = 0.82
    _scripted_hover_z_above: float = 0.20
    _topple_park_y: float = 0.06
    _topple_park_x_tol: float = 0.03
    _topple_park_z_tol: float = 0.02

    _topple_push_trigger: float = 0.055
    _topple_push_dist: float = 0.08
    _topple_push_pos_step: float = 0.010
    _topple_push_accel_step: float = 0.004
    _topple_push_stop_cos: float = 0.60
    _topple_push_max_steps: int = 30

    _topple_retreat_z_above: float = 0.25

    @property
    def _scripted_rider(self) -> Actor:
        return self.block

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._block_half_y

    @property
    def _scripted_socket_chamfer(self) -> float:
        return 0.0

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, torch.Tensor]:
        wait_z = deck_z + self._scripted_hover_z_above
        park_z = deck_z + self._block_contact_frac * self._block_height
        return wait_z, park_z

    def _scripted_hold_gripper(self) -> torch.Tensor:
        _, close_val = self._scripted_gripper_values()
        return torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)

    def _scripted_park_y(self, oy: float, sign: torch.Tensor, block_pos: torch.Tensor) -> torch.Tensor:
        return oy + self._topple_park_y * sign

    def _scripted_block_up_dot(self) -> torch.Tensor:
        return quaternion_to_matrix(self.block.pose.q)[:, 2, 2].abs()

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_topple_phase"):
            self._topple_phase = torch.full((b,), self._PHASE_PARK, dtype=torch.long, device=device)
            self._topple_push_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._topple_push_x = torch.zeros((b,), device=device)
            self._topple_push_y = torch.zeros((b,), device=device)
            self._topple_retreat_xy = torch.zeros((b, 2), device=device)
            self._topple_push_prev_delta = torch.zeros((b, 3), device=device)
        if env_idx is None:
            self._topple_phase[:] = self._PHASE_PARK
            self._topple_push_counter[:] = 0
            self._topple_push_prev_delta[:] = 0.0
        else:
            self._topple_phase[env_idx] = self._PHASE_PARK
            self._topple_push_counter[env_idx] = 0
            self._topple_push_prev_delta[env_idx] = 0.0

    def _topple_push_position_input(
        self, target_world: torch.Tensor, tcp_pos: torch.Tensor, pushing: torch.Tensor
    ) -> torch.Tensor:
        delta = target_world - tcp_pos
        accel = self._topple_push_accel_step
        prev = self._topple_push_prev_delta
        delta = prev + (delta - prev).clamp(-accel, accel)
        cap = self._topple_push_pos_step
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap / xy_norm, max=1.0)
        z = delta[:, 2:3].clamp(-cap, cap)
        clamped = torch.cat([xy, z], dim=-1)
        self._topple_push_prev_delta = torch.where(pushing.unsqueeze(-1), clamped, torch.zeros_like(clamped))
        return torch.clip(self._apply_reach_guard(clamped, tcp_pos) / 0.1, -1.0, 1.0)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_topple_phase"):
            self.reset_scripted()

        device = self.device
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        block_pos = self.block.pose.p
        self._update_auto_speed_scale(tcp_pos, block_pos)

        sign = self.belt.direction_sign
        motion_dir = torch.zeros((self.num_envs, 2), device=device)
        motion_dir[:, 1] = sign
        self._scripted_motion_dir = motion_dir

        _hover_z, park_z = self._scripted_wait_lower_z(deck_z, block_pos)
        park_y = self._scripted_park_y(oy, sign, block_pos)
        up_dot = self._scripted_block_up_dot()

        phase = self._topple_phase

        gap = (tcp_pos[:, 1] - block_pos[:, 1]) * sign
        in_lane = (torch.abs(tcp_pos[:, 0] - block_pos[:, 0]) < self._topple_park_x_tol) & (
            torch.abs(tcp_pos[:, 2] - park_z) < self._topple_park_z_tol
        )
        to_push = (phase == self._PHASE_PARK) & in_lane & (gap < self._topple_push_trigger)
        self._topple_push_x = torch.where(to_push, block_pos[:, 0], self._topple_push_x)
        self._topple_push_y = torch.where(to_push, tcp_pos[:, 1] - self._topple_push_dist * sign, self._topple_push_y)
        phase = torch.where(to_push, torch.full_like(phase, self._PHASE_PUSH), phase)

        in_push = phase == self._PHASE_PUSH
        self._topple_push_counter = torch.where(
            in_push,
            self._topple_push_counter + 1,
            torch.zeros_like(self._topple_push_counter),
        )
        to_retreat = in_push & (
            (up_dot < self._topple_push_stop_cos) | (self._topple_push_counter >= self._topple_push_max_steps)
        )
        self._topple_retreat_xy = torch.where(to_retreat.unsqueeze(-1), tcp_pos[:, :2], self._topple_retreat_xy)
        phase = torch.where(to_retreat, torch.full_like(phase, self._PHASE_RETREAT), phase)

        self._topple_phase = phase

        target = tcp_pos.clone()

        is_park = phase == self._PHASE_PARK
        target[:, 0] = torch.where(is_park, block_pos[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_park, park_y, target[:, 1])
        target[:, 2] = torch.where(is_park, park_z, target[:, 2])

        is_push = phase == self._PHASE_PUSH
        target[:, 0] = torch.where(is_push, self._topple_push_x, target[:, 0])
        target[:, 1] = torch.where(is_push, self._topple_push_y, target[:, 1])
        target[:, 2] = torch.where(is_push, park_z, target[:, 2])

        is_retreat = phase == self._PHASE_RETREAT
        target[:, 0] = torch.where(is_retreat, self._topple_retreat_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_retreat, self._topple_retreat_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(
            is_retreat,
            torch.full_like(target[:, 2], deck_z + self._topple_retreat_z_above),
            target[:, 2],
        )

        approach_pos = self._position_input(target, tcp_pos)
        push_pos = self._topple_push_position_input(target, tcp_pos, is_push)
        pos_input = torch.where(is_push.unsqueeze(-1), push_pos, approach_pos)

        rot_input = self._rotation_input(tcp_q)
        gripper = self._scripted_hold_gripper()

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
