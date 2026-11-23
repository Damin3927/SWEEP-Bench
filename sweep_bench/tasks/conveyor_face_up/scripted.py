from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion, quaternion_multiply

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)
from sweep_bench.tasks.conveyor_face_up.assets import face_normal_world

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedFaceUpMixin(_ScriptedGraspFamilyMixin):
    die: Actor
    pad: Actor
    _die_half: float
    _pad_half_z: float
    _command_sign: torch.Tensor
    _commanded_face: torch.Tensor

    _scripted_approach_z_above: float = 0.15
    _scripted_close_xy_tol: float = 0.0265
    _scripted_lift_z_above: float = 0.16
    _scripted_hold_z_above: float = 0.12
    _scripted_pos_step = 0.014
    _scripted_v_ee_max = 0.21

    _fu_grasp_reach_y: float = 0.36

    _FU_GRASPING = 0
    _FU_SECURE = 1
    _FU_CARRY = 2
    _FU_LOWER = 3
    _FU_RELEASE = 4
    _FU_RETREAT = 5

    _fu_transport_pos_step: float = 0.045
    _fu_transport_accel_step: float = 0.006

    _fu_carry_z_above: float = 0.12
    _fu_secure_settle_speed: float = 0.15
    _fu_secure_max_steps: int = 20
    _fu_grasp_check_dist: float = 0.06

    _fu_lower_xy_tol: float = 0.02
    _fu_roll_done_cos: float = math.cos(math.radians(12.0))
    _fu_servo_start_cos: float = math.cos(math.radians(35.0))

    _fu_lower_pos_step: float = 0.012
    _fu_lower_aim_gap: float = 0.020
    _fu_release_gap: float = 0.030
    _fu_release_steps: int = 10
    _fu_retreat_z_above: float = 0.20
    _fu_retreat_back: float = 0.06

    _fu_jaw_sign: float = 1.0

    @property
    def _scripted_rider(self) -> Actor:
        return self.die

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_fu_phase"):
            self._fu_phase = torch.full((b,), self._FU_GRASPING, dtype=torch.long, device=device)
            self._fu_secure_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._fu_release_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._fu_secure_xy = torch.zeros((b, 2), device=device)
            self._fu_prev_delta = torch.zeros((b, 3), device=device)
        if env_idx is None:
            self._fu_phase[:] = self._FU_GRASPING
            self._fu_secure_counter[:] = 0
            self._fu_release_counter[:] = 0
            self._fu_prev_delta[:] = 0.0
        else:
            self._fu_phase[env_idx] = self._FU_GRASPING
            self._fu_secure_counter[env_idx] = 0
            self._fu_release_counter[env_idx] = 0
            self._fu_prev_delta[env_idx] = 0.0

    def _face_up_correction_q(self) -> torch.Tensor:
        axis = face_normal_world(self.die.pose.q, self._commanded_face)
        up = torch.zeros_like(axis)
        up[:, 2] = 1.0
        v = torch.linalg.cross(axis, up, dim=-1)
        sin_t = torch.linalg.norm(v, dim=-1)
        theta = torch.atan2(sin_t, axis[:, 2].clamp(-1.0, 1.0))
        n = v / sin_t.clamp(min=1e-9).unsqueeze(-1)
        half = 0.5 * theta
        return torch.cat([torch.cos(half).unsqueeze(-1), n * torch.sin(half).unsqueeze(-1)], dim=-1)

    def _fu_base_target_q(self) -> torch.Tensor:
        device = self.device
        b = self.num_envs
        y_body = torch.zeros((b, 3), device=device)
        y_body[:, 0] = self._fu_jaw_sign
        z_body = torch.zeros((b, 3), device=device)
        z_body[:, 2] = -1.0
        x_body = torch.linalg.cross(y_body, z_body, dim=-1)
        R = torch.stack([x_body, y_body, z_body], dim=-1)
        return matrix_to_quaternion(R)

    def _gripper_down_target_q(self) -> torch.Tensor:
        base_q = self._fu_base_target_q()
        if not hasattr(self, "_fu_phase"):
            return base_q
        tcp_q = cast("Panda", self.agent).tcp.pose.q
        half = 0.25 * math.pi
        roll_q = torch.zeros_like(base_q)
        roll_q[:, 0] = math.cos(half)
        roll_q[:, 1] = math.sin(half) * self._command_sign
        rolled = quaternion_multiply(roll_q, base_q)
        servoed = quaternion_multiply(self._face_up_correction_q(), tcp_q)

        face_cos = face_normal_world(self.die.pose.q, self._commanded_face)[:, 2]
        rolling = (self._fu_phase == self._FU_CARRY) | (self._fu_phase == self._FU_LOWER)
        target = torch.where(rolling.unsqueeze(-1), rolled, base_q)
        target = torch.where((rolling & (face_cos > self._fu_servo_start_cos)).unsqueeze(-1), servoed, target)
        frozen = (self._fu_phase == self._FU_RELEASE) | (self._fu_phase == self._FU_RETREAT)
        return torch.where(frozen.unsqueeze(-1), tcp_q, target)

    def _position_input(self, target_world: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
        oy = self.workspace_offset[1]
        reach = self._fu_grasp_reach_y
        y = target_world[:, 1:2].clamp(oy - reach, oy + reach)
        return super()._position_input(torch.cat([target_world[:, :1], y, target_world[:, 2:]], dim=-1), tcp_pos)

    def _transport_position_input(
        self, target_world: torch.Tensor, tcp_pos: torch.Tensor, ramping: torch.Tensor, cap: torch.Tensor
    ) -> torch.Tensor:
        delta = target_world - tcp_pos
        accel = self._fu_transport_accel_step
        prev = self._fu_prev_delta
        delta = prev + (delta - prev).clamp(-accel, accel)
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap / xy_norm, max=1.0)
        z = delta[:, 2:3].clamp(-cap, cap)
        clamped = torch.cat([xy, z], dim=-1)
        self._fu_prev_delta = torch.where(ramping.unsqueeze(-1), clamped, torch.zeros_like(clamped))
        return torch.clip(clamped / 0.1, -1.0, 1.0)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_fu_phase"):
            self.reset_scripted()

        grasp_action = super().compute_scripted_action()

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        die_pos = self.die.pose.p
        pad_pos = self.pad.pose.p
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        pad_top = pad_pos[:, 2] + self._pad_half_z
        carry_z = deck_z + self._fu_carry_z_above

        fu = self._fu_phase

        grasp_done = self._scripted_phase == self._PHASE_HOLD
        in_hand = torch.linalg.norm(tcp_pos - die_pos, dim=-1) < self._fu_grasp_check_dist
        to_secure = (fu == self._FU_GRASPING) & grasp_done & in_hand
        self._fu_secure_xy = torch.where(to_secure.unsqueeze(-1), tcp_pos[:, :2], self._fu_secure_xy)
        fu = torch.where(to_secure, torch.full_like(fu, self._FU_SECURE), fu)

        in_secure = fu == self._FU_SECURE
        self._fu_secure_counter = torch.where(
            in_secure, self._fu_secure_counter + 1, torch.zeros_like(self._fu_secure_counter)
        )
        agent_speed = torch.linalg.norm(agent.tcp.linear_velocity, dim=-1)
        settled_arm = (agent_speed < self._fu_secure_settle_speed) | (
            self._fu_secure_counter >= self._fu_secure_max_steps
        )
        fu = torch.where(in_secure & settled_arm, torch.full_like(fu, self._FU_CARRY), fu)

        face_cos = face_normal_world(self.die.pose.q, self._commanded_face)[:, 2]
        over_pad = torch.linalg.norm(tcp_pos[:, :2] - pad_pos[:, :2], dim=-1) < self._fu_lower_xy_tol
        to_lower = (fu == self._FU_CARRY) & over_pad & (face_cos > self._fu_roll_done_cos)
        fu = torch.where(to_lower, torch.full_like(fu, self._FU_LOWER), fu)

        die_bottom = die_pos[:, 2] - self._die_half
        to_release = (fu == self._FU_LOWER) & ((die_bottom - pad_top) < self._fu_release_gap)
        fu = torch.where(to_release, torch.full_like(fu, self._FU_RELEASE), fu)

        in_release = fu == self._FU_RELEASE
        self._fu_release_counter = torch.where(
            in_release, self._fu_release_counter + 1, torch.zeros_like(self._fu_release_counter)
        )
        fu = torch.where(
            in_release & (self._fu_release_counter >= self._fu_release_steps),
            torch.full_like(fu, self._FU_RETREAT),
            fu,
        )

        self._fu_phase = fu

        target = tcp_pos.clone()

        is_secure = fu == self._FU_SECURE
        target[:, 0] = torch.where(is_secure, self._fu_secure_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_secure, self._fu_secure_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_secure, torch.full_like(target[:, 2], carry_z), target[:, 2])

        is_carry = fu == self._FU_CARRY
        target[:, 0] = torch.where(is_carry, pad_pos[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_carry, pad_pos[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_carry, torch.full_like(target[:, 2], carry_z), target[:, 2])

        is_lower = fu == self._FU_LOWER
        lower_z = pad_top + self._die_half + self._fu_lower_aim_gap
        target[:, 0] = torch.where(is_lower, pad_pos[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_lower, pad_pos[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_lower, lower_z, target[:, 2])

        is_retreat = fu == self._FU_RETREAT
        target[:, 0] = torch.where(is_retreat, pad_pos[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_retreat, pad_pos[:, 1] - self._fu_retreat_back * self._command_sign, target[:, 1])
        target[:, 2] = torch.where(is_retreat, pad_top + self._fu_retreat_z_above, target[:, 2])

        in_transport = fu != self._FU_GRASPING
        cap = torch.where(
            (fu == self._FU_LOWER).unsqueeze(-1),
            torch.full_like(tcp_pos[:, :1], self._fu_lower_pos_step),
            torch.full_like(tcp_pos[:, :1], self._fu_transport_pos_step),
        )
        pos_input = self._transport_position_input(target, tcp_pos, in_transport, cap)
        rot_input = self._rotation_input(tcp_q)
        open_val, close_val = self._scripted_gripper_values()
        is_open = (fu == self._FU_RELEASE) | (fu == self._FU_RETREAT)
        gripper = torch.where(
            is_open,
            torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32),
            torch.full((self.num_envs,), close_val, device=self.device, dtype=torch.float32),
        ).unsqueeze(-1)
        transport_action = torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)

        return torch.where(in_transport.unsqueeze(-1), transport_action, grasp_action)
