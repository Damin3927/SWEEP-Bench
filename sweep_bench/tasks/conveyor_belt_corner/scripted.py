from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedBeltCornerMixin(_ScriptedGraspFamilyMixin):
    part: Actor
    _part_half_z: torch.Tensor
    _release_xy: torch.Tensor

    _scripted_approach_z_above: float = 0.15
    _scripted_lift_z_above: float = 0.18
    _scripted_hold_z_above: float = 0.14
    _scripted_pos_step: float = 0.014
    _scripted_v_ee_max: float = 0.21

    _BC_GRASPING = 0
    _BC_CARRY = 1
    _BC_LOWER = 2
    _BC_RELEASE = 3
    _BC_RETREAT = 4

    _bc_transport_pos_step: float = 0.045

    _bc_carry_z_above: float = 0.14
    _bc_retreat_z_above: float = 0.22
    _bc_over_target_xy_tol: float = 0.02
    _bc_lower_press: float = 0.012
    _bc_release_gap: float = 0.006
    _bc_release_steps: int = 10
    _bc_grasp_check_dist: float = 0.06
    _bc_handoff_settle_speed: float = 0.15
    _bc_handoff_max_steps: int = 8

    @property
    def _scripted_rider(self) -> Actor:
        return self.part

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_bc_phase"):
            self._bc_phase = torch.full((b,), self._BC_GRASPING, dtype=torch.long, device=device)
            self._bc_release_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._bc_hold_counter = torch.zeros((b,), dtype=torch.long, device=device)
        if env_idx is None:
            self._bc_phase[:] = self._BC_GRASPING
            self._bc_release_counter[:] = 0
            self._bc_hold_counter[:] = 0
        else:
            self._bc_phase[env_idx] = self._BC_GRASPING
            self._bc_release_counter[env_idx] = 0
            self._bc_hold_counter[env_idx] = 0

    def _transport_position_input(self, target_world: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
        delta = target_world - tcp_pos
        cap = self._bc_transport_pos_step
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap / xy_norm, max=1.0)
        z = delta[:, 2:3].clamp(-cap, cap)
        return torch.clip(torch.cat([xy, z], dim=-1) / 0.1, -1.0, 1.0)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_bc_phase"):
            self.reset_scripted()

        grasp_action = super().compute_scripted_action()

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        part_pos = self.part.pose.p

        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        carry_z = deck_z + self._bc_carry_z_above
        release_centre_z = deck_z + self._part_half_z

        bc = self._bc_phase

        at_hold = self._scripted_phase == self._PHASE_HOLD
        part_in_hand = torch.linalg.norm(tcp_pos - part_pos, dim=-1) < self._bc_grasp_check_dist
        self._bc_hold_counter = torch.where(at_hold, self._bc_hold_counter + 1, torch.zeros_like(self._bc_hold_counter))
        calm = (torch.linalg.norm(agent.tcp.linear_velocity, dim=-1) < self._bc_handoff_settle_speed) | (
            self._bc_hold_counter >= self._bc_handoff_max_steps
        )
        to_carry = (bc == self._BC_GRASPING) & at_hold & part_in_hand & calm
        bc = torch.where(to_carry, torch.full_like(bc, self._BC_CARRY), bc)

        over_target = torch.linalg.norm(tcp_pos[:, :2] - self._release_xy, dim=-1) < self._bc_over_target_xy_tol
        to_lower = (bc == self._BC_CARRY) & over_target
        bc = torch.where(to_lower, torch.full_like(bc, self._BC_LOWER), bc)

        part_bottom = part_pos[:, 2] - self._part_half_z
        to_release = (bc == self._BC_LOWER) & ((part_bottom - deck_z) < self._bc_release_gap)
        bc = torch.where(to_release, torch.full_like(bc, self._BC_RELEASE), bc)

        in_release = bc == self._BC_RELEASE
        self._bc_release_counter = torch.where(
            in_release, self._bc_release_counter + 1, torch.zeros_like(self._bc_release_counter)
        )
        to_retreat = in_release & (self._bc_release_counter >= self._bc_release_steps)
        bc = torch.where(to_retreat, torch.full_like(bc, self._BC_RETREAT), bc)

        self._bc_phase = bc

        target = tcp_pos.clone()

        is_carry = bc == self._BC_CARRY
        target[:, 0] = torch.where(is_carry, self._release_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_carry, self._release_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_carry, torch.full_like(target[:, 2], carry_z), target[:, 2])

        is_lower = bc == self._BC_LOWER
        target[:, 0] = torch.where(is_lower, self._release_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_lower, self._release_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_lower, release_centre_z - self._bc_lower_press, target[:, 2])

        is_retreat = bc == self._BC_RETREAT
        target[:, 2] = torch.where(
            is_retreat, torch.full_like(target[:, 2], deck_z + self._bc_retreat_z_above), target[:, 2]
        )

        in_transport = bc != self._BC_GRASPING
        pos_input = self._transport_position_input(target, tcp_pos)
        rot_input = self._rotation_input(tcp_q)

        open_val, close_val = self._scripted_gripper_values()
        is_open = (bc == self._BC_RELEASE) | (bc == self._BC_RETREAT)
        gripper = torch.where(
            is_open,
            torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32),
            torch.full((self.num_envs,), close_val, device=self.device, dtype=torch.float32),
        ).unsqueeze(-1)
        transport_action = torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)

        return torch.where(in_transport.unsqueeze(-1), transport_action, grasp_action)
