from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedDecantMixin(_ScriptedGraspFamilyMixin):
    cup: Actor
    bin: Actor
    _bin_rim_local_z: float

    _scripted_approach_z_above: float = 0.18
    _scripted_close_xy_tol: float = 0.020
    _scripted_close_z_tol: float = 0.016
    _scripted_lift_z_above: float = 0.16
    _scripted_hold_z_above: float = 0.12

    _dc_lift_pos_step: float = 0.018
    _dc_lift_accel_step: float = 0.006

    _DC_GRASPING = 0
    _DC_SECURE = 1
    _DC_CARRY = 2
    _DC_TIP = 3
    _DC_SETTLE = 4

    _dc_transport_pos_step: float = 0.045
    _dc_transport_accel_step: float = 0.006
    _dc_carry_z_above: float = 0.12
    _dc_secure_settle_speed: float = 0.15
    _dc_secure_max_steps: int = 20
    _dc_belt_clear_dx: float = 0.14
    _dc_carry_tip_angle: float = math.radians(60.0)
    _dc_tip_angle: float = math.radians(115.0)
    _dc_tip_xy_tol: float = 0.04
    _dc_carry_max_steps: int = 110
    _dc_tip_done_slack: float = math.radians(8.0)
    _dc_tip_max_steps: int = 90
    _dc_pour_z_above_rim: float = 0.12
    _dc_tip_lead: float = 0.05
    _dc_grasp_check_dist: float = 0.06

    _dc_approach_reach_lo: float = 0.66
    _dc_approach_reach_hi: float = 0.74
    _dc_approach_bearing_lo: float = 0.35
    _dc_approach_bearing_hi: float = 0.70
    _dc_approach_hold_r_cap: float = 0.62
    _dc_reach_slow_lo: float = 0.66
    _dc_reach_slow_hi: float = 0.74
    _dc_reach_slowdown: float = 0.85

    @property
    def _scripted_rider(self) -> Actor:
        return self.cup

    def _gripper_down_target_q(self) -> torch.Tensor:
        q_down = super()._gripper_down_target_q()
        if not hasattr(self, "_dc_tip_cmd"):
            return q_down
        half = 0.5 * self._dc_tip_cmd
        q_roll = torch.zeros_like(q_down)
        q_roll[:, 0] = torch.cos(half)
        q_roll[:, 2] = torch.sin(half)
        return quaternion_multiply(q_down, q_roll)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_dc_stage"):
            self._dc_stage = torch.full((b,), self._DC_GRASPING, dtype=torch.long, device=device)
            self._dc_secure_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._dc_carry_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._dc_tip_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._dc_tip_cmd = torch.zeros((b,), device=device)
            self._dc_secure_xy = torch.zeros((b, 2), device=device)
            self._dc_prev_delta = torch.zeros((b, 3), device=device)
            self._dc_lift_prev_delta = torch.zeros((b, 3), device=device)
        if env_idx is None:
            self._dc_stage[:] = self._DC_GRASPING
            self._dc_secure_counter[:] = 0
            self._dc_carry_counter[:] = 0
            self._dc_tip_counter[:] = 0
            self._dc_tip_cmd[:] = 0.0
            self._dc_prev_delta[:] = 0.0
            self._dc_lift_prev_delta[:] = 0.0
        else:
            self._dc_stage[env_idx] = self._DC_GRASPING
            self._dc_secure_counter[env_idx] = 0
            self._dc_carry_counter[env_idx] = 0
            self._dc_tip_counter[env_idx] = 0
            self._dc_tip_cmd[env_idx] = 0.0
            self._dc_prev_delta[env_idx] = 0.0
            self._dc_lift_prev_delta[env_idx] = 0.0

    def _transport_position_input(
        self, target_world: torch.Tensor, tcp_pos: torch.Tensor, ramping: torch.Tensor
    ) -> torch.Tensor:
        delta = target_world - tcp_pos
        accel = self._dc_transport_accel_step
        prev = self._dc_prev_delta
        delta = prev + (delta - prev).clamp(-accel, accel)
        base_xy = self.agent.robot.pose.p[:, :2]
        r_tcp = torch.linalg.norm(tcp_pos[:, :2] - base_xy, dim=-1)
        near_reach = ((r_tcp - self._dc_reach_slow_lo) / (self._dc_reach_slow_hi - self._dc_reach_slow_lo)).clamp(
            0.0, 1.0
        )
        cap = (self._dc_transport_pos_step * (1.0 - self._dc_reach_slowdown * near_reach)).unsqueeze(-1)
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap / xy_norm, max=1.0)
        z = torch.maximum(torch.minimum(delta[:, 2:3], cap), -cap)
        clamped = torch.cat([xy, z], dim=-1)
        self._dc_prev_delta = torch.where(ramping.unsqueeze(-1), clamped, torch.zeros_like(clamped))
        return torch.clip(clamped / 0.1, -1.0, 1.0)

    def _lift_position_input(
        self, target_world: torch.Tensor, tcp_pos: torch.Tensor, gentle: torch.Tensor
    ) -> torch.Tensor:
        prev = self._dc_lift_prev_delta
        accel = self._dc_lift_accel_step
        delta = prev + ((target_world - tcp_pos) - prev).clamp(-accel, accel)
        cap = self._dc_lift_pos_step
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap / xy_norm, max=1.0)
        z = delta[:, 2:3].clamp(-cap, cap)
        clamped = torch.cat([xy, z], dim=-1)
        self._dc_lift_prev_delta = torch.where(gentle.unsqueeze(-1), clamped, self._scripted_prev_delta)
        return torch.clip(clamped / 0.1, -1.0, 1.0)

    def _aim_approach_xy(self, tcp_xy: torch.Tensor, aim_xy: torch.Tensor) -> torch.Tensor:
        base_xy = self.agent.robot.pose.p[:, :2]
        d_grasp = self._dc_secure_xy - base_xy
        d_aim = aim_xy - base_xy
        d_tcp = tcp_xy - base_xy
        th_grasp = torch.atan2(d_grasp[:, 1], d_grasp[:, 0])
        th_aim = torch.atan2(d_aim[:, 1], d_aim[:, 0])
        th_tcp = torch.atan2(d_tcp[:, 1], d_tcp[:, 0])
        r_aim = torch.linalg.norm(d_aim, dim=-1)
        r_grasp = torch.linalg.norm(d_grasp, dim=-1)
        th_span = th_grasp - th_aim
        frac = torch.where(
            th_span.abs() > 1e-3,
            ((th_tcp - th_aim) / th_span).clamp(0.0, 1.0),
            torch.zeros_like(th_span),
        )
        radius_reach = (
            (r_grasp - self._dc_approach_reach_lo) / (self._dc_approach_reach_hi - self._dc_approach_reach_lo)
        ).clamp(0.0, 1.0)
        bearing_reach = (
            (th_span.abs() - self._dc_approach_bearing_lo)
            / (self._dc_approach_bearing_hi - self._dc_approach_bearing_lo)
        ).clamp(0.0, 1.0)
        reach = torch.maximum(radius_reach, bearing_reach)
        comfortable = r_grasp < self._dc_approach_reach_lo
        r_hold = torch.where(comfortable, r_grasp, torch.clamp(r_grasp, max=self._dc_approach_hold_r_cap))
        r_target = r_aim + (r_hold - r_aim) * frac * reach
        heading = torch.stack([torch.cos(th_aim), torch.sin(th_aim)], dim=-1)
        return base_xy + r_target.unsqueeze(-1) * heading

    def _tip_aim_xy(self) -> torch.Tensor:
        bin_xy = self.bin.pose.p[:, :2]
        return bin_xy + self._dc_tip_lead * self._scripted_motion_dir

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_dc_stage"):
            self.reset_scripted()

        grasp_action = super().compute_scripted_action()

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        cup_pos = self.cup.pose.p
        bin_pos = self.bin.pose.p

        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        carry_z = deck_z + self._dc_carry_z_above
        pour_z = bin_pos[:, 2] + self._bin_rim_local_z + self._dc_pour_z_above_rim

        st = self._dc_stage
        ox = self.workspace_offset[0]
        belt_cleared = tcp_pos[:, 0] < ox - self._dc_belt_clear_dx
        aim_xy = self._tip_aim_xy()

        gentle = (self._scripted_phase == self._PHASE_LIFT) | (self._scripted_phase == self._PHASE_HOLD)
        lift_target = tcp_pos.clone()
        lift_target[:, 2] = deck_z + self._scripted_lift_z_above
        lift_pos_input = self._lift_position_input(lift_target, tcp_pos, gentle)
        grasp_action = torch.cat(
            [torch.where(gentle.unsqueeze(-1), lift_pos_input, grasp_action[:, :3]), grasp_action[:, 3:]], dim=-1
        )

        agent_speed = torch.linalg.norm(agent.tcp.linear_velocity, dim=-1)

        grasp_done = self._scripted_phase == self._PHASE_HOLD
        cup_in_hand = torch.linalg.norm(tcp_pos - cup_pos, dim=-1) < self._dc_grasp_check_dist
        to_secure = (st == self._DC_GRASPING) & grasp_done & cup_in_hand
        self._dc_secure_xy = torch.where(to_secure.unsqueeze(-1), tcp_pos[:, :2], self._dc_secure_xy)
        st = torch.where(to_secure, torch.full_like(st, self._DC_SECURE), st)

        in_secure = st == self._DC_SECURE
        self._dc_secure_counter = torch.where(
            in_secure, self._dc_secure_counter + 1, torch.zeros_like(self._dc_secure_counter)
        )
        settled_arm = (agent_speed < self._dc_secure_settle_speed) | (
            self._dc_secure_counter >= self._dc_secure_max_steps
        )
        st = torch.where(in_secure & settled_arm, torch.full_like(st, self._DC_CARRY), st)

        in_carry = st == self._DC_CARRY
        self._dc_carry_counter = torch.where(
            in_carry, self._dc_carry_counter + 1, torch.zeros_like(self._dc_carry_counter)
        )
        at_aim = torch.linalg.norm(tcp_pos[:, :2] - aim_xy, dim=-1) < self._dc_tip_xy_tol
        to_tip = in_carry & ((at_aim & belt_cleared) | (self._dc_carry_counter >= self._dc_carry_max_steps))
        st = torch.where(to_tip, torch.full_like(st, self._DC_TIP), st)

        in_tip = st == self._DC_TIP
        self._dc_tip_counter = torch.where(in_tip, self._dc_tip_counter + 1, torch.zeros_like(self._dc_tip_counter))
        cup_up_cos = 1.0 - 2.0 * (self.cup.pose.q[:, 1] ** 2 + self.cup.pose.q[:, 2] ** 2)
        rolled = cup_up_cos < math.cos(self._dc_tip_angle - self._dc_tip_done_slack)
        to_settle = in_tip & (rolled | (self._dc_tip_counter >= self._dc_tip_max_steps))
        st = torch.where(to_settle, torch.full_like(st, self._DC_SETTLE), st)

        self._dc_stage = st

        pre_roll = (st == self._DC_CARRY) & belt_cleared
        full_roll = (st == self._DC_TIP) | (st == self._DC_SETTLE)
        tip_cmd = torch.zeros_like(self._dc_tip_cmd)
        tip_cmd = torch.where(pre_roll, torch.full_like(tip_cmd, self._dc_carry_tip_angle), tip_cmd)
        self._dc_tip_cmd = torch.where(full_roll, torch.full_like(tip_cmd, self._dc_tip_angle), tip_cmd)

        target = tcp_pos.clone()

        is_secure = st == self._DC_SECURE
        target[:, 0] = torch.where(is_secure, self._dc_secure_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_secure, self._dc_secure_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_secure, torch.full_like(target[:, 2], carry_z), target[:, 2])

        approach_xy = self._aim_approach_xy(tcp_pos[:, :2], aim_xy)
        target[:, 0] = torch.where(in_carry, approach_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(in_carry, approach_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(in_carry, torch.full_like(target[:, 2], carry_z), target[:, 2])

        pouring = (st == self._DC_TIP) | (st == self._DC_SETTLE)
        target[:, 0] = torch.where(pouring, aim_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(pouring, aim_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(pouring, pour_z, target[:, 2])

        in_transport = st != self._DC_GRASPING
        pos_input = self._transport_position_input(target, tcp_pos, in_transport)
        rot_input = self._rotation_input(tcp_q)

        _open_val, close_val = self._scripted_gripper_values()
        gripper = torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)
        transport_action = torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)

        return torch.where(in_transport.unsqueeze(-1), transport_action, grasp_action)
