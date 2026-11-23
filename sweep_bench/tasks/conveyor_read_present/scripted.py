from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_matrix,
)

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedReadPresentMixin(_ScriptedGraspFamilyMixin):
    part: Actor
    _present_xyz: torch.Tensor
    _target_heading: torch.Tensor
    _place_xy: torch.Tensor
    _part_half: torch.Tensor
    _read_latched: torch.Tensor

    _scripted_approach_z_above: float = 0.15
    _scripted_lift_z_above: float = 0.16
    _scripted_hold_z_above: float = 0.12
    _scripted_pos_step = 0.014
    _scripted_v_ee_max = 0.21

    _RP_GRASPING = 0
    _RP_SECURE = 1
    _RP_CARRY = 2
    _RP_DWELL = 3
    _RP_RETURN = 4
    _RP_LOWER = 5
    _RP_RELEASE = 6
    _RP_RETREAT = 7

    _rp_transport_pos_step: float = 0.045
    _rp_transport_accel_step: float = 0.006
    _rp_secure_settle_speed: float = 0.15
    _rp_secure_max_steps: int = 20
    _rp_grasp_check_dist: float = 0.06
    _rp_present_xy_tol: float = 0.030
    _rp_present_z_tol: float = 0.030
    _rp_carry_max_steps: int = 90
    _rp_dwell_max_steps: int = 60

    _rp_return_z_above: float = 0.10
    _rp_place_xy_tol: float = 0.02
    _rp_return_max_steps: int = 80
    _rp_lower_press: float = 0.012
    _rp_release_gap: float = 0.006
    _rp_lower_max_steps: int = 60
    _rp_release_steps: int = 3
    _rp_retreat_z_above: float = 0.20

    _rp_hold_gripper: float = -0.5

    @property
    def _scripted_rider(self) -> Actor:
        return self.part

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_rp_phase"):
            self._rp_phase = torch.full((b,), self._RP_GRASPING, dtype=torch.long, device=device)
            self._rp_stage_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._rp_secure_xy = torch.zeros((b, 2), device=device)
            self._rp_frozen_q = torch.zeros((b, 4), device=device)
            self._rp_frozen_q[:, 0] = 1.0
            self._rp_prev_delta = torch.zeros((b, 3), device=device)
        if env_idx is None:
            self._rp_phase[:] = self._RP_GRASPING
            self._rp_stage_counter[:] = 0
            self._rp_prev_delta[:] = 0.0
        else:
            self._rp_phase[env_idx] = self._RP_GRASPING
            self._rp_stage_counter[env_idx] = 0
            self._rp_prev_delta[env_idx] = 0.0

    def _mark_heading_error(self) -> torch.Tensor:
        mark = quaternion_to_matrix(self.part.pose.q)[:, :, 0]
        mark_angle = torch.atan2(mark[:, 1], mark[:, 0])
        err = self._target_heading - mark_angle
        return torch.remainder(err + torch.pi, 2.0 * torch.pi) - torch.pi

    def _gripper_down_target_q(self) -> torch.Tensor:
        base_q = super()._gripper_down_target_q()
        if not hasattr(self, "_rp_phase") or not hasattr(self, "_target_heading"):
            return base_q
        tcp_q = cast("Panda", self.agent).tcp.pose.q
        rel = quaternion_to_matrix(quaternion_multiply(tcp_q, quaternion_invert(base_q)))
        wrist_yaw = torch.atan2(rel[:, 1, 0], rel[:, 0, 0])
        half = 0.5 * (wrist_yaw + self._mark_heading_error())
        q_yaw = torch.zeros_like(base_q)
        q_yaw[:, 0] = torch.cos(half)
        q_yaw[:, 3] = torch.sin(half)
        yawed_q = quaternion_multiply(q_yaw, base_q)

        servoing = (self._rp_phase >= self._RP_CARRY) & (self._rp_phase <= self._RP_LOWER)
        self._rp_frozen_q = torch.where(servoing.unsqueeze(-1), yawed_q, self._rp_frozen_q)
        target = torch.where(servoing.unsqueeze(-1), yawed_q, base_q)
        return torch.where((self._rp_phase >= self._RP_RELEASE).unsqueeze(-1), self._rp_frozen_q, target)

    def _transport_position_input(
        self, target_world: torch.Tensor, tcp_pos: torch.Tensor, in_chain: torch.Tensor, limited: torch.Tensor
    ) -> torch.Tensor:
        delta = target_world - tcp_pos
        accel = self._rp_transport_accel_step
        prev = self._rp_prev_delta
        eased = prev + (delta - prev).clamp(-accel, accel)
        delta = torch.where(limited.unsqueeze(-1), eased, delta)
        cap = self._rp_transport_pos_step
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap / xy_norm, max=1.0)
        z = delta[:, 2:3].clamp(-cap, cap)
        clamped = torch.cat([xy, z], dim=-1)
        self._rp_prev_delta = torch.where(in_chain.unsqueeze(-1), clamped, torch.zeros_like(clamped))
        return torch.clip(clamped / 0.1, -1.0, 1.0)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_rp_phase"):
            self.reset_scripted()

        grasp_action = super().compute_scripted_action()

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        part_pos = self.part.pose.p
        _ox, _oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        secure_z = deck_z + self._scripted_lift_z_above
        return_z = deck_z + self._rp_return_z_above
        rest_centre_z = deck_z + self._part_half[:, 2]

        rp = self._rp_phase
        entered = torch.zeros_like(rp, dtype=torch.bool)
        held = self._rp_stage_counter

        grasp_done = self._scripted_phase == self._PHASE_HOLD
        part_in_hand = torch.linalg.norm(tcp_pos - part_pos, dim=-1) < self._rp_grasp_check_dist
        to_secure = (rp == self._RP_GRASPING) & grasp_done & part_in_hand
        self._rp_secure_xy = torch.where(to_secure.unsqueeze(-1), tcp_pos[:, :2], self._rp_secure_xy)
        rp = torch.where(to_secure, torch.full_like(rp, self._RP_SECURE), rp)
        entered |= to_secure

        agent_speed = torch.linalg.norm(agent.tcp.linear_velocity, dim=-1)
        settled_arm = (agent_speed < self._rp_secure_settle_speed) | (held >= self._rp_secure_max_steps)
        to_carry = (rp == self._RP_SECURE) & settled_arm & ~entered
        rp = torch.where(to_carry, torch.full_like(rp, self._RP_CARRY), rp)
        entered |= to_carry

        at_present = (
            torch.linalg.norm(tcp_pos[:, :2] - self._present_xyz[:, :2], dim=-1) < self._rp_present_xy_tol
        ) & (torch.abs(tcp_pos[:, 2] - self._present_xyz[:, 2]) < self._rp_present_z_tol)
        to_dwell = (rp == self._RP_CARRY) & (at_present | (held >= self._rp_carry_max_steps)) & ~entered
        rp = torch.where(to_dwell, torch.full_like(rp, self._RP_DWELL), rp)
        entered |= to_dwell

        read_done = self._read_latched if hasattr(self, "_read_latched") else torch.zeros_like(entered)
        to_return = (rp == self._RP_DWELL) & (read_done | (held >= self._rp_dwell_max_steps)) & ~entered
        rp = torch.where(to_return, torch.full_like(rp, self._RP_RETURN), rp)
        entered |= to_return

        over_place = torch.linalg.norm(tcp_pos[:, :2] - self._place_xy, dim=-1) < self._rp_place_xy_tol
        to_lower = (rp == self._RP_RETURN) & (over_place | (held >= self._rp_return_max_steps)) & ~entered
        rp = torch.where(to_lower, torch.full_like(rp, self._RP_LOWER), rp)
        entered |= to_lower

        part_bottom_gap = part_pos[:, 2] - self._part_half[:, 2] - deck_z
        to_release = (
            (rp == self._RP_LOWER)
            & ((part_bottom_gap < self._rp_release_gap) | (held >= self._rp_lower_max_steps))
            & ~entered
        )
        rp = torch.where(to_release, torch.full_like(rp, self._RP_RELEASE), rp)
        entered |= to_release

        to_retreat = (rp == self._RP_RELEASE) & (held >= self._rp_release_steps) & ~entered
        rp = torch.where(to_retreat, torch.full_like(rp, self._RP_RETREAT), rp)
        entered |= to_retreat

        self._rp_phase = rp
        self._rp_stage_counter = torch.where(entered, torch.zeros_like(held), held + 1)

        target = tcp_pos.clone()

        is_secure = rp == self._RP_SECURE
        target[:, 0] = torch.where(is_secure, self._rp_secure_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_secure, self._rp_secure_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_secure, torch.full_like(target[:, 2], secure_z), target[:, 2])

        is_presenting = (rp == self._RP_CARRY) | (rp == self._RP_DWELL)
        target = torch.where(is_presenting.unsqueeze(-1), self._present_xyz, target)

        is_return = rp == self._RP_RETURN
        target[:, 0] = torch.where(is_return, self._place_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_return, self._place_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_return, torch.full_like(target[:, 2], return_z), target[:, 2])

        is_lower = rp == self._RP_LOWER
        target[:, 0] = torch.where(is_lower, self._place_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_lower, self._place_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_lower, rest_centre_z - self._rp_lower_press, target[:, 2])

        target[:, 2] = torch.where(
            rp == self._RP_RETREAT, torch.full_like(target[:, 2], deck_z + self._rp_retreat_z_above), target[:, 2]
        )

        in_chain = rp != self._RP_GRASPING
        limited = (rp == self._RP_CARRY) | is_return
        pos_input = self._transport_position_input(target, tcp_pos, in_chain, limited)
        rot_input = self._rotation_input(tcp_q)

        open_val, _close_val = self._scripted_gripper_values()
        let_go = rp >= self._RP_RELEASE
        gripper = torch.where(
            let_go,
            torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32),
            torch.full((self.num_envs,), self._rp_hold_gripper, device=self.device, dtype=torch.float32),
        ).unsqueeze(-1)
        chain_action = torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)

        return torch.where(in_chain.unsqueeze(-1), chain_action, grasp_action)
