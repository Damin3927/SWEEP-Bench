from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedCullFlawedMixin(_ScriptedGraspFamilyMixin):
    part_flawed: Actor
    _bin_pos: torch.Tensor
    _is_flawed: torch.Tensor
    _part_half_z: torch.Tensor
    _part_half_fp: torch.Tensor
    _bin_inner_half_x: float
    _bin_inner_half_y: float
    _bin_wall_half_height: float
    _bin_floor_half_height: float

    _scripted_lift_z_above: float = 0.16
    _scripted_hold_z_above: float = 0.12
    _scripted_approach_z_above: float = 0.15
    _scripted_pos_step = 0.014
    _scripted_v_ee_max = 0.21

    _PP_GRASPING = 0
    _PP_SECURE = 1
    _PP_CARRY = 2
    _PP_LOWER = 3
    _PP_RELEASE = 4
    _PP_RETREAT = 5

    _pp_transport_pos_step: float = 0.045
    _pp_transport_accel_step: float = 0.006
    _pp_carry_z_above: float = 0.12
    _pp_secure_settle_speed: float = 0.15
    _pp_secure_max_steps: int = 20
    _pp_belt_clear_dx: float = 0.14
    _pp_rim_clearance: float = 0.02
    _pp_approach_reach_lo: float = 0.66
    _pp_approach_reach_hi: float = 0.74
    _pp_approach_bearing_lo: float = 0.35
    _pp_approach_bearing_hi: float = 0.70
    _pp_approach_hold_r_cap: float = 0.62
    _pp_reach_slow_lo: float = 0.66
    _pp_reach_slow_hi: float = 0.74
    _pp_reach_slowdown: float = 0.85
    _pp_grasp_check_dist: float = 0.06
    _pp_lower_below_floor: float = 0.03
    _scripted_release_gap: float = 0.02
    _scripted_release_steps: int = 10
    _pp_drop_zone_r: float = 0.035

    @property
    def _scripted_rider(self) -> Actor:
        return self.part_flawed

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_pp_phase"):
            self._pp_phase = torch.full((b,), self._PP_GRASPING, dtype=torch.long, device=device)
            self._pp_release_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._pp_secure_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._pp_secure_xy = torch.zeros((b, 2), device=device)
            self._pp_prev_delta = torch.zeros((b, 3), device=device)
        if env_idx is None:
            self._pp_phase[:] = self._PP_GRASPING
            self._pp_release_counter[:] = 0
            self._pp_secure_counter[:] = 0
            self._pp_prev_delta[:] = 0.0
        else:
            self._pp_phase[env_idx] = self._PP_GRASPING
            self._pp_release_counter[env_idx] = 0
            self._pp_secure_counter[env_idx] = 0
            self._pp_prev_delta[env_idx] = 0.0

    def _transport_position_input(
        self, target_world: torch.Tensor, tcp_pos: torch.Tensor, ramping: torch.Tensor
    ) -> torch.Tensor:
        delta = target_world - tcp_pos
        accel = self._pp_transport_accel_step
        prev = self._pp_prev_delta
        delta = prev + (delta - prev).clamp(-accel, accel)
        base_xy = self.agent.robot.pose.p[:, :2]
        r_tcp = torch.linalg.norm(tcp_pos[:, :2] - base_xy, dim=-1)
        near_reach = ((r_tcp - self._pp_reach_slow_lo) / (self._pp_reach_slow_hi - self._pp_reach_slow_lo)).clamp(
            0.0, 1.0
        )
        cap = (self._pp_transport_pos_step * (1.0 - self._pp_reach_slowdown * near_reach)).unsqueeze(-1)
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap / xy_norm, max=1.0)
        z = torch.maximum(torch.minimum(delta[:, 2:3], cap), -cap)
        clamped = torch.cat([xy, z], dim=-1)
        self._pp_prev_delta = torch.where(ramping.unsqueeze(-1), clamped, torch.zeros_like(clamped))
        return torch.clip(clamped / 0.1, -1.0, 1.0)

    def _bin_approach_xy(self, tcp_xy: torch.Tensor, bin_xy: torch.Tensor) -> torch.Tensor:
        base_xy = self.agent.robot.pose.p[:, :2]
        d_grasp = self._pp_secure_xy - base_xy
        d_bin = bin_xy - base_xy
        d_tcp = tcp_xy - base_xy
        th_grasp = torch.atan2(d_grasp[:, 1], d_grasp[:, 0])
        th_bin = torch.atan2(d_bin[:, 1], d_bin[:, 0])
        th_tcp = torch.atan2(d_tcp[:, 1], d_tcp[:, 0])
        r_bin = torch.linalg.norm(d_bin, dim=-1)
        r_grasp = torch.linalg.norm(d_grasp, dim=-1)
        th_span = th_grasp - th_bin
        frac = torch.where(
            th_span.abs() > 1e-3,
            ((th_tcp - th_bin) / th_span).clamp(0.0, 1.0),
            torch.zeros_like(th_span),
        )
        radius_reach = (
            (r_grasp - self._pp_approach_reach_lo) / (self._pp_approach_reach_hi - self._pp_approach_reach_lo)
        ).clamp(0.0, 1.0)
        bearing_reach = (
            (th_span.abs() - self._pp_approach_bearing_lo)
            / (self._pp_approach_bearing_hi - self._pp_approach_bearing_lo)
        ).clamp(0.0, 1.0)
        reach = torch.maximum(radius_reach, bearing_reach)
        comfortable = r_grasp < self._pp_approach_reach_lo
        r_hold = torch.where(comfortable, r_grasp, torch.clamp(r_grasp, max=self._pp_approach_hold_r_cap))
        r_target = r_bin + (r_hold - r_bin) * frac * reach
        heading = torch.stack([torch.cos(th_bin), torch.sin(th_bin)], dim=-1)
        return base_xy + r_target.unsqueeze(-1) * heading

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_pp_phase"):
            self.reset_scripted()

        grasp_action = super().compute_scripted_action()

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        part_pos = self.part_flawed.pose.p
        bin_pos = self._bin_pos

        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        carry_z = deck_z + self._pp_carry_z_above
        bin_floor_top = bin_pos[:, 2] + self._bin_floor_half_height

        pp = self._pp_phase

        agent_speed = torch.linalg.norm(agent.tcp.linear_velocity, dim=-1)

        grasp_done = self._scripted_phase == self._PHASE_HOLD
        part_in_hand = torch.linalg.norm(tcp_pos - part_pos, dim=-1) < self._pp_grasp_check_dist
        to_secure = (pp == self._PP_GRASPING) & grasp_done & part_in_hand
        self._pp_secure_xy = torch.where(to_secure.unsqueeze(-1), tcp_pos[:, :2], self._pp_secure_xy)
        pp = torch.where(to_secure, torch.full_like(pp, self._PP_SECURE), pp)

        in_secure = pp == self._PP_SECURE
        self._pp_secure_counter = torch.where(
            in_secure,
            self._pp_secure_counter + 1,
            torch.zeros_like(self._pp_secure_counter),
        )
        settled_arm = (agent_speed < self._pp_secure_settle_speed) | (
            self._pp_secure_counter >= self._pp_secure_max_steps
        )
        pp = torch.where(in_secure & settled_arm, torch.full_like(pp, self._PP_CARRY), pp)

        ox = self.workspace_offset[0]
        belt_cleared = tcp_pos[:, 0] < ox - self._pp_belt_clear_dx
        pp = torch.where((pp == self._PP_CARRY) & belt_cleared, torch.full_like(pp, self._PP_LOWER), pp)

        part_bottom = part_pos[:, 2] - self._part_half_z
        near_floor = (part_bottom - bin_floor_top) < self._scripted_release_gap
        pp = torch.where((pp == self._PP_LOWER) & near_floor, torch.full_like(pp, self._PP_RELEASE), pp)

        in_release = pp == self._PP_RELEASE
        self._pp_release_counter = torch.where(
            in_release,
            self._pp_release_counter + 1,
            torch.zeros_like(self._pp_release_counter),
        )
        pp = torch.where(
            in_release & (self._pp_release_counter >= self._scripted_release_steps),
            torch.full_like(pp, self._PP_RETREAT),
            pp,
        )

        self._pp_phase = pp

        target = tcp_pos.clone()

        is_secure = pp == self._PP_SECURE
        target[:, 0] = torch.where(is_secure, self._pp_secure_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_secure, self._pp_secure_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_secure, torch.full_like(target[:, 2], carry_z), target[:, 2])

        approach_xy = self._bin_approach_xy(tcp_pos[:, :2], bin_pos[:, :2])

        is_carry = pp == self._PP_CARRY
        target[:, 0] = torch.where(is_carry, approach_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_carry, approach_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_carry, torch.full_like(target[:, 2], carry_z), target[:, 2])

        is_lower = pp == self._PP_LOWER
        lower_z = bin_floor_top - self._pp_lower_below_floor
        rim_top = bin_pos[:, 2] + self._bin_floor_half_height + 2.0 * self._bin_wall_half_height
        rim_hover_z = rim_top + self._part_half_z + self._pp_rim_clearance
        rel_part = part_pos[:, :2] - bin_pos[:, :2]
        in_drop_zone = torch.linalg.norm(rel_part, dim=-1) < (self._pp_drop_zone_r - self._part_half_fp).clamp(min=0.01)
        guarded_z = torch.where(in_drop_zone, lower_z, rim_hover_z)
        lower_xy = torch.where(
            self._scripted_belt_stopped().unsqueeze(-1),
            approach_xy - (part_pos[:, :2] - tcp_pos[:, :2]),
            approach_xy,
        )
        target[:, 0] = torch.where(is_lower, lower_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_lower, lower_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_lower, guarded_z, target[:, 2])

        is_retreat = pp == self._PP_RETREAT
        target[:, 2] = torch.where(is_retreat, torch.full_like(target[:, 2], carry_z), target[:, 2])

        in_transport = pp != self._PP_GRASPING
        pos_input = self._transport_position_input(target, tcp_pos, in_transport)
        rot_input = self._rotation_input(tcp_q)

        open_val, close_val = self._scripted_gripper_values()
        is_open = (pp == self._PP_RELEASE) | (pp == self._PP_RETREAT)
        gripper = torch.where(
            is_open,
            torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32),
            torch.full((self.num_envs,), close_val, device=self.device, dtype=torch.float32),
        ).unsqueeze(-1)
        transport_action = torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)

        cull_action = torch.where(in_transport.unsqueeze(-1), transport_action, grasp_action)

        idle_action = torch.zeros_like(cull_action)
        idle_action[:, 6] = open_val
        return torch.where(self._is_flawed.unsqueeze(-1), cull_action, idle_action)
