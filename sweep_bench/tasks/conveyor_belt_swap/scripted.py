from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedBeltLoadMixin(_ScriptedGraspFamilyMixin):
    puck: Actor
    _puck_half_z: torch.Tensor
    _release_xy: torch.Tensor
    _release_z_offset: torch.Tensor

    _scripted_lift_z_above: float = 0.16
    _scripted_hold_z_above: float = 0.12

    _BL_GRASPING = 0
    _BL_CARRY = 1
    _BL_LOWER = 2
    _BL_RELEASE = 3
    _BL_RETREAT = 4

    _bl_transport_pos_step: float = 0.045
    _bl_transport_accel_step: float | None = None
    _bl_carry_z_above: float = 0.12
    _bl_retreat_z_above: float = 0.20
    _bl_over_target_xy_tol: float = 0.02
    _bl_lower_press: float = 0.012
    _bl_release_gap: float = 0.006
    _bl_release_steps: int = 10
    _bl_grasp_check_dist: float = 0.06

    @property
    def _scripted_rider(self) -> Actor:
        return self.puck

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_bl_phase"):
            self._bl_phase = torch.full((b,), self._BL_GRASPING, dtype=torch.long, device=device)
            self._bl_release_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._bl_prev_delta = torch.zeros((b, 3), device=device)
        if env_idx is None:
            self._bl_phase[:] = self._BL_GRASPING
            self._bl_release_counter[:] = 0
            self._bl_prev_delta[:] = 0.0
        else:
            self._bl_phase[env_idx] = self._BL_GRASPING
            self._bl_release_counter[env_idx] = 0
            self._bl_prev_delta[env_idx] = 0.0

    def _transport_position_input(
        self, target_world: torch.Tensor, tcp_pos: torch.Tensor, ramping: torch.Tensor
    ) -> torch.Tensor:
        delta = target_world - tcp_pos
        accel = self._bl_transport_accel_step
        prev = self._bl_prev_delta
        if accel is not None:
            delta = prev + (delta - prev).clamp(-accel, accel)
        cap = self._bl_transport_pos_step
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap / xy_norm, max=1.0)
        z = delta[:, 2:3].clamp(-cap, cap)
        clamped = torch.cat([xy, z], dim=-1)
        self._bl_prev_delta = torch.where(ramping.unsqueeze(-1), clamped, torch.zeros_like(clamped))
        return torch.clip(self._apply_reach_guard(clamped, tcp_pos) / 0.1, -1.0, 1.0)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_bl_phase"):
            self.reset_scripted()

        grasp_action = super().compute_scripted_action()

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        puck_pos = self.puck.pose.p

        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        carry_z = deck_z + self._bl_carry_z_above
        release_centre_z = deck_z + self._puck_half_z + self._release_z_offset

        bl = self._bl_phase

        grasp_done = self._scripted_phase == self._PHASE_HOLD
        puck_in_hand = torch.linalg.norm(tcp_pos - puck_pos, dim=-1) < self._bl_grasp_check_dist
        to_carry = (bl == self._BL_GRASPING) & grasp_done & puck_in_hand
        bl = torch.where(to_carry, torch.full_like(bl, self._BL_CARRY), bl)

        over_target = torch.linalg.norm(tcp_pos[:, :2] - self._release_xy, dim=-1) < self._bl_over_target_xy_tol
        to_lower = (bl == self._BL_CARRY) & over_target
        bl = torch.where(to_lower, torch.full_like(bl, self._BL_LOWER), bl)

        puck_bottom = puck_pos[:, 2] - self._puck_half_z
        at_release_height = (puck_bottom - deck_z) < (self._release_z_offset + self._bl_release_gap)
        to_release = (bl == self._BL_LOWER) & at_release_height
        bl = torch.where(to_release, torch.full_like(bl, self._BL_RELEASE), bl)

        in_release = bl == self._BL_RELEASE
        self._bl_release_counter = torch.where(
            in_release,
            self._bl_release_counter + 1,
            torch.zeros_like(self._bl_release_counter),
        )
        to_retreat = in_release & (self._bl_release_counter >= self._bl_release_steps)
        bl = torch.where(to_retreat, torch.full_like(bl, self._BL_RETREAT), bl)

        self._bl_phase = bl

        target = tcp_pos.clone()

        is_carry = bl == self._BL_CARRY
        target[:, 0] = torch.where(is_carry, self._release_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_carry, self._release_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_carry, torch.full_like(target[:, 2], carry_z), target[:, 2])

        is_lower = bl == self._BL_LOWER
        target[:, 0] = torch.where(is_lower, self._release_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_lower, self._release_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_lower, release_centre_z - self._bl_lower_press, target[:, 2])

        is_retreat = bl == self._BL_RETREAT
        target[:, 2] = torch.where(
            is_retreat, torch.full_like(target[:, 2], deck_z + self._bl_retreat_z_above), target[:, 2]
        )

        in_transport = bl != self._BL_GRASPING
        pos_input = self._transport_position_input(target, tcp_pos, in_transport)
        rot_input = self._rotation_input(tcp_q)

        open_val, close_val = self._scripted_gripper_values()
        is_open = (bl == self._BL_RELEASE) | (bl == self._BL_RETREAT)
        gripper = torch.where(
            is_open,
            torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32),
            torch.full((self.num_envs,), close_val, device=self.device, dtype=torch.float32),
        ).unsqueeze(-1)
        transport_action = torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)

        return torch.where(in_transport.unsqueeze(-1), transport_action, grasp_action)


class _HandoffSettleGraspMixin(_ScriptedGraspFamilyMixin):
    _bs_handoff_settle_speed: float = 0.15
    _bs_handoff_max_steps: int = 8

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_bs_settled"):
            self._bs_settled = torch.zeros((b,), dtype=torch.bool, device=device)
            self._bs_hold_counter = torch.zeros((b,), dtype=torch.long, device=device)
        if env_idx is None:
            self._bs_settled[:] = False
            self._bs_hold_counter[:] = 0
        else:
            self._bs_settled[env_idx] = False
            self._bs_hold_counter[env_idx] = 0

    def compute_scripted_action(self) -> torch.Tensor:
        if not hasattr(self, "_bs_settled"):
            self.reset_scripted()
        action = super().compute_scripted_action()

        at_hold = self._scripted_phase == self._PHASE_HOLD
        self._bs_hold_counter = torch.where(at_hold, self._bs_hold_counter + 1, torch.zeros_like(self._bs_hold_counter))
        agent = cast("Panda", self.agent)
        calm = (torch.linalg.norm(agent.tcp.linear_velocity, dim=-1) < self._bs_handoff_settle_speed) | (
            self._bs_hold_counter >= self._bs_handoff_max_steps
        )
        self._bs_settled = self._bs_settled | (at_hold & calm)
        self._scripted_phase = torch.where(
            at_hold & ~self._bs_settled,
            torch.full_like(self._scripted_phase, self._PHASE_LIFT),
            self._scripted_phase,
        )
        return action


class ScriptedBeltSwapMixin(ScriptedBeltLoadMixin, _HandoffSettleGraspMixin):
    _scripted_approach_z_above: float = 0.15
    _scripted_close_lead_time: float = 0.08
    _scripted_pos_step: float = 0.01
    _scripted_v_ee_max: float = 0.15
