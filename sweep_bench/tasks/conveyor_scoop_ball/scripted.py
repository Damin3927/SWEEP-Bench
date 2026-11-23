from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@dataclass(frozen=True)
class _AimPoint:
    pose: Pose


class ScriptedScoopMixin(_ScriptedInsertFamilyMixin):
    _PHASE_STAGE = 0
    _PHASE_CAPTURE = 1
    _PHASE_LIFT = 2
    _PHASE_HOLD = 3

    ball: Actor
    _scoop_below_tcp: float
    _scoop_grip_lead: float
    _scoop_capture_lead: torch.Tensor
    _scoop_ball_radius: torch.Tensor

    _scripted_accel_step: float | None = 0.006

    _scripted_comove_lead_time: float = 0.0

    _scoop_dip_radius: float = 0.03
    _scoop_dip_ramp: float = 0.03

    _scripted_settle_cos: float = 0.99

    _scoop_seat_clearance: float = -0.002
    _scoop_wait_clearance: float = 0.085

    _scoop_capture_closing: float = 0.30
    _scoop_push_through: float = 0.012
    _scoop_capture_scale_floor: float = 4.0

    _scripted_lift_tcp_z_above: float = 0.185
    _scripted_hold_tcp_z_above: float = 0.165
    _scoop_lift_y_max: float = 0.30
    _scoop_lift_retreat_speed: float = 0.22
    _scoop_lift_accel: float = 0.8
    _scoop_lift_z_speed: float = 0.25
    _scoop_lift_pitch: float = 0.17

    _scoop_seat_engage: float = 0.004
    _scoop_stage_xy_tol: float = 0.035

    _scoop_phase: torch.Tensor
    _scoop_lead: torch.Tensor

    @property
    def _scripted_rider(self) -> Actor:
        sign = self.belt.direction_sign
        p = self.ball.pose.p.clone()
        p[:, 1] = p[:, 1] + sign * (self._scoop_grip_lead + self._scoop_lead)
        return cast("Actor", _AimPoint(Pose.create_from_pq(p)))

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._scoop_dip_radius

    @property
    def _scripted_socket_chamfer(self) -> float:
        return self._scoop_dip_ramp

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, torch.Tensor]:
        wait_z = deck_z + self._scoop_wait_clearance + self._scoop_below_tcp
        seated_z = deck_z + self._scoop_seat_clearance + self._scoop_below_tcp

        tcp_q = cast("Panda", self.agent).tcp.pose.q
        rot_cos = torch.abs((tcp_q * self._gripper_down_target_q()).sum(dim=-1))
        lower_z = torch.where(
            rot_cos > self._scripted_settle_cos,
            torch.full_like(rot_cos, seated_z),
            torch.full_like(rot_cos, wait_z),
        )
        return wait_z, lower_z

    def _scripted_hold_gripper(self) -> torch.Tensor:
        _, close_val = self._scripted_gripper_values()
        return torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)

    def _gripper_down_target_q(self) -> torch.Tensor:
        base = super()._gripper_down_target_q()
        if not hasattr(self, "_scoop_phase"):
            return base
        half = 0.5 * self._scoop_lift_pitch
        pitch = torch.zeros_like(base)
        pitch[:, 0] = math.cos(half)
        pitch[:, 2] = -math.sin(half)
        identity = torch.zeros_like(base)
        identity[:, 0] = 1.0
        loaded = (self._scoop_phase >= self._PHASE_LIFT).unsqueeze(-1)
        return quaternion_multiply(base, torch.where(loaded, pitch, identity))

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        if not hasattr(self, "_scoop_phase"):
            self._scoop_phase = torch.full((self.num_envs,), self._PHASE_STAGE, dtype=torch.long, device=self.device)
            self._scoop_lead = torch.zeros((self.num_envs,), device=self.device)
            self._scoop_hold = torch.zeros((self.num_envs, 3), device=self.device)
            self._scoop_goal = torch.zeros((self.num_envs, 3), device=self.device)
            self._scoop_hold_v = torch.zeros((self.num_envs,), device=self.device)
        self._reset_scripted_shared(env_idx)
        if env_idx is None:
            self._scoop_phase[:] = self._PHASE_STAGE
            self._scoop_lead[:] = self._scoop_capture_lead
            self._scoop_hold[:] = 0.0
            self._scoop_goal[:] = 0.0
            self._scoop_hold_v[:] = 0.0
        else:
            self._scoop_phase[env_idx] = self._PHASE_STAGE
            self._scoop_lead[env_idx] = self._scoop_capture_lead[env_idx]
            self._scoop_hold[env_idx] = 0.0
            self._scoop_goal[env_idx] = 0.0
            self._scoop_hold_v[env_idx] = 0.0

    def _advance_capture_lead(self, tcp_pos: torch.Tensor) -> None:
        step = self._scoop_capture_closing / self.control_freq
        sign = self.belt.direction_sign
        gap_now = (tcp_pos[:, 1] - self.ball.pose.p[:, 1]) * sign - self._scoop_grip_lead
        closing = self._scoop_phase == self._PHASE_CAPTURE
        walked = torch.minimum(self._scoop_lead - step, gap_now).clamp(min=-self._scoop_push_through)
        self._scoop_lead = torch.where(closing, walked, self._scoop_lead)

    def _advance_hold_target(
        self,
        tcp_pos: torch.Tensor,
        tcp_vel: torch.Tensor,
        deck_z: float,
        entering: torch.Tensor,
        capture_aim: torch.Tensor,
    ) -> torch.Tensor:
        oy = self.workspace_offset[1]
        goal = tcp_pos.clone()
        goal[:, 1] = goal[:, 1].clamp(oy - self._scoop_lift_y_max, oy + self._scoop_lift_y_max)
        goal[:, 2] = deck_z + self._scripted_lift_tcp_z_above
        latch = entering.unsqueeze(-1)
        start = tcp_pos.clone()
        start[:, 1] = torch.where(self._scripted_belt_stopped(), capture_aim[:, 1], tcp_pos[:, 1])
        self._scoop_goal = torch.where(latch, goal, self._scoop_goal)
        self._scoop_hold = torch.where(latch, start, self._scoop_hold)
        self._scoop_hold_v = torch.where(entering, tcp_vel[:, 1], self._scoop_hold_v)

        dt = 1.0 / self.control_freq
        v_des = ((self._scoop_goal[:, 1] - self._scoop_hold[:, 1]) / dt).clamp(
            -self._scoop_lift_retreat_speed, self._scoop_lift_retreat_speed
        )
        dv = self._scoop_lift_accel * dt
        self._scoop_hold_v = self._scoop_hold_v + (v_des - self._scoop_hold_v).clamp(-dv, dv)
        self._scoop_hold[:, 1] = self._scoop_hold[:, 1] + self._scoop_hold_v * dt
        z_step = self._scoop_lift_z_speed * dt
        self._scoop_hold[:, 2] = self._scoop_hold[:, 2] + (self._scoop_goal[:, 2] - self._scoop_hold[:, 2]).clamp(
            -z_step, z_step
        )
        return self._scoop_hold

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_scoop_phase"):
            self.reset_scripted()

        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        self._advance_capture_lead(agent.tcp.pose.p)

        state = self._intercept_state()
        phase = self._scoop_phase

        seated = state.tcp_pos[:, 2] - (deck_z + self._scoop_below_tcp) < self._scoop_seat_engage
        staged = (phase == self._PHASE_STAGE) & seated & (state.xy_dist < self._scoop_stage_xy_tol)
        phase = torch.where(staged, torch.full_like(phase, self._PHASE_CAPTURE), phase)

        loaded = (phase == self._PHASE_CAPTURE) & self._ball_in_pocket()
        phase = torch.where(loaded, torch.full_like(phase, self._PHASE_LIFT), phase)

        risen = (phase == self._PHASE_LIFT) & (state.tcp_pos[:, 2] > deck_z + self._scripted_hold_tcp_z_above)
        phase = torch.where(risen, torch.full_like(phase, self._PHASE_HOLD), phase)
        entering_lift = (phase >= self._PHASE_LIFT) & (self._scoop_phase < self._PHASE_LIFT)
        self._scoop_phase = phase

        if self._scripted_auto_scale is not None:
            floored = (phase == self._PHASE_CAPTURE) | (self._scripted_belt_stopped() & (phase >= self._PHASE_LIFT))
            self._scripted_auto_scale = torch.where(
                floored,
                self._scripted_auto_scale.clamp(min=self._scoop_capture_scale_floor),
                self._scripted_auto_scale,
            )

        target_world = state.target_world.clone()
        target_world[:, 2] = torch.where(
            phase == self._PHASE_CAPTURE,
            torch.full_like(target_world[:, 2], deck_z + self._scoop_seat_clearance + self._scoop_below_tcp),
            target_world[:, 2],
        )

        lifting = (phase >= self._PHASE_LIFT).unsqueeze(-1)
        lift_target = self._advance_hold_target(
            state.tcp_pos, agent.tcp.linear_velocity, deck_z, entering_lift, state.target_world
        )
        target_world = torch.where(lifting, lift_target, target_world)

        pos_input = self._position_input(target_world, state.tcp_pos)
        rot_input = self._rotation_input(state.tcp_q)
        gripper = self._scripted_hold_gripper()
        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)

    def _ball_in_pocket(self) -> torch.Tensor:
        raise NotImplementedError
