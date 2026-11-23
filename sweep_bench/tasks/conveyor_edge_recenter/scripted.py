from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedEdgeRecenterMixin(_ScriptedGraspFamilyMixin):
    part: Actor
    _rest_half_z: torch.Tensor

    _scripted_approach_z_above: float = 0.15
    _scripted_lift_z_above: float = 0.12
    _scripted_hold_z_above: float = 0.09

    _ER_GRASPING = 0
    _ER_SHIFT = 1
    _ER_LOWER = 2
    _ER_RELEASE = 3
    _ER_RETREAT = 4

    _er_transport_pos_step: float = 0.045
    _er_carry_z_above: float = 0.09
    _er_retreat_z_above: float = 0.22
    _er_over_target_xy_tol: float = 0.012
    _er_lower_press: float = 0.010
    _er_release_gap: float = 0.005
    _er_release_steps: int = 6
    _er_grasp_check_dist: float = 0.07
    _er_setdown_y_max: float = 0.0
    _er_reseat_x_offset: float = 0.0

    @property
    def _scripted_rider(self) -> Actor:
        return self.part

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_er_phase"):
            self._er_phase = torch.full((b,), self._ER_GRASPING, dtype=torch.long, device=device)
            self._er_release_counter = torch.zeros((b,), dtype=torch.long, device=device)
        if env_idx is None:
            self._er_phase[:] = self._ER_GRASPING
            self._er_release_counter[:] = 0
        else:
            self._er_phase[env_idx] = self._ER_GRASPING
            self._er_release_counter[env_idx] = 0

    def _reseat_position_input(self, target_world: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
        delta = target_world - tcp_pos
        cap = self._er_transport_pos_step
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap / xy_norm, max=1.0)
        z = delta[:, 2:3].clamp(-cap, cap)
        return torch.clip(self._apply_reach_guard(torch.cat([xy, z], dim=-1), tcp_pos) / 0.1, -1.0, 1.0)

    def _setdown_y(self, tcp_pos: torch.Tensor) -> torch.Tensor:
        oy = self.workspace_offset[1]
        sign = self.belt.direction_sign
        signed = (tcp_pos[:, 1] - oy) * sign
        moving = torch.abs(self.belt.velocity) > 1e-6
        signed = torch.where(moving, signed.clamp(max=self._er_setdown_y_max), signed)
        return oy + sign * signed

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_er_phase"):
            self.reset_scripted()

        grasp_action = super().compute_scripted_action()

        ox = self.workspace_offset[0]
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        part_pos = self.part.pose.p

        er = self._er_phase

        grasp_done = self._scripted_phase == self._PHASE_HOLD
        part_in_hand = torch.linalg.norm(tcp_pos - part_pos, dim=-1) < self._er_grasp_check_dist
        to_shift = (er == self._ER_GRASPING) & grasp_done & part_in_hand
        er = torch.where(to_shift, torch.full_like(er, self._ER_SHIFT), er)

        setdown_y = self._setdown_y(tcp_pos)
        target_xy = torch.stack([torch.full_like(setdown_y, ox + self._er_reseat_x_offset), setdown_y], dim=-1)

        over_target = torch.linalg.norm(tcp_pos[:, :2] - target_xy, dim=-1) < self._er_over_target_xy_tol
        to_lower = (er == self._ER_SHIFT) & over_target
        er = torch.where(to_lower, torch.full_like(er, self._ER_LOWER), er)

        part_bottom = part_pos[:, 2] - self._rest_half_z
        to_release = (er == self._ER_LOWER) & ((part_bottom - deck_z) < self._er_release_gap)
        er = torch.where(to_release, torch.full_like(er, self._ER_RELEASE), er)

        in_release = er == self._ER_RELEASE
        self._er_release_counter = torch.where(
            in_release,
            self._er_release_counter + 1,
            torch.zeros_like(self._er_release_counter),
        )
        to_retreat = in_release & (self._er_release_counter >= self._er_release_steps)
        er = torch.where(to_retreat, torch.full_like(er, self._ER_RETREAT), er)

        self._er_phase = er

        target = tcp_pos.clone()

        is_shift = er == self._ER_SHIFT
        target[:, 0] = torch.where(is_shift, target_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_shift, target_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(
            is_shift, torch.full_like(target[:, 2], deck_z + self._er_carry_z_above), target[:, 2]
        )

        is_lower = er == self._ER_LOWER
        target[:, 0] = torch.where(is_lower, target_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_lower, target_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_lower, deck_z + self._rest_half_z - self._er_lower_press, target[:, 2])

        is_retreat = er == self._ER_RETREAT
        target[:, 2] = torch.where(
            is_retreat, torch.full_like(target[:, 2], deck_z + self._er_retreat_z_above), target[:, 2]
        )

        in_reseat = er != self._ER_GRASPING
        pos_input = self._reseat_position_input(target, tcp_pos)
        rot_input = self._rotation_input(tcp_q)

        open_val, close_val = self._scripted_gripper_values()
        is_open = (er == self._ER_RELEASE) | (er == self._ER_RETREAT)
        gripper = torch.where(
            is_open,
            torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32),
            torch.full((self.num_envs,), close_val, device=self.device, dtype=torch.float32),
        ).unsqueeze(-1)
        reseat_action = torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)

        return torch.where(in_reseat.unsqueeze(-1), reseat_action, grasp_action)
