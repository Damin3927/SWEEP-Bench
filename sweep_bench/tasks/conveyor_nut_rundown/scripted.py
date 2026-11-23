from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor

    from sweep_bench.tasks.conveyor_base.scripted_families import _InterceptState

_TWO_PI = 2.0 * math.pi


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, _TWO_PI) - math.pi


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    t = x.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class _AimPoint:
    pose: Pose


class ScriptedNutRunDownMixin(_ScriptedInsertFamilyMixin):
    _PHASE_STROKE = 0
    _PHASE_BACKOFF = 1

    fixture: Actor
    _wing_contact_radius: float
    _nut_half_height: float
    _nut_outer_radius: float
    _wing_count: int

    _finger_below_tcp: float = 0.009

    _scripted_hover_z_above: float = 0.20
    _scripted_pos_step: float = 0.018
    _rd_wait_clearance: float = 0.03
    _rd_commit_radius: float = 0.012
    _rd_commit_ramp: float = 0.012
    _rd_arm_xy_tol: float = 0.030
    _rd_arm_z_tol: float = 0.006
    _rd_engage_y: float = 0.22
    _rd_engage_lead_time: float = 0.6
    _rd_wait_z_above: float = 0.12

    _rd_approach_lag: float = 1.571
    _rd_arc_gap: float = 0.60
    _rd_arc_gap_min: float = 0.45
    _rd_wing_lead_max: float = 0.60
    _rd_slow_band: float = 0.35
    _rd_radius_floor: float = 0.019

    _rd_stroke_margin: float = math.radians(12.0)
    _rd_stop_time: float = 0.05
    _rd_retract_radius: float = 0.11
    _rd_withdraw_z_above: float = 0.13
    _rd_stroke_budget: int = 45
    _rd_max_retries: int = 2
    _rd_backoff_floor: float = math.radians(6.0)
    _rd_stall_speed: float = 0.05
    _rd_work_step: float = 0.018
    _rd_retreat_step: float = 0.03
    _rd_vertical_step: float = 0.004
    _rd_work_boost: float = 0.014
    _rd_stall_ramp: float = 8.0
    _rd_stall_eps: float = math.radians(0.5)
    _rd_stall_give_up: float = 16.0

    def _nut_yaw(self) -> torch.Tensor:
        raise NotImplementedError

    def _nut_height(self) -> torch.Tensor:
        raise NotImplementedError

    def _nut_spin(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def _turn_target(self) -> torch.Tensor:
        raise NotImplementedError

    _rundown_turn: torch.Tensor

    def _fixture_to_world(self, local: torch.Tensor) -> torch.Tensor:
        rot = quaternion_to_matrix(self.fixture.pose.q)
        return self.fixture.pose.p + torch.bmm(rot, local.unsqueeze(-1)).squeeze(-1)

    def _world_to_fixture(self, world: torch.Tensor) -> torch.Tensor:
        rot = quaternion_to_matrix(self.fixture.pose.q)
        return torch.bmm(rot.transpose(1, 2), (world - self.fixture.pose.p).unsqueeze(-1)).squeeze(-1)

    def _rd_contact_local_z(self) -> torch.Tensor:
        return self._nut_height() + self._finger_below_tcp

    def _rd_arc_point(self, phi: torch.Tensor, local_z: torch.Tensor) -> torch.Tensor:
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 0] = -self._wing_contact_radius * torch.sin(phi)
        local[:, 1] = self._wing_contact_radius * torch.cos(phi)
        local[:, 2] = local_z
        return self._fixture_to_world(local)

    def _rd_wrap_wing(self, delta: torch.Tensor) -> torch.Tensor:
        pitch = _TWO_PI / self._wing_count
        return torch.remainder(delta + 0.5 * pitch, pitch) - 0.5 * pitch

    def _rd_pick_wing(self) -> torch.Tensor:
        raw = self._nut_yaw()
        base_local = self._world_to_fixture(self.agent.robot.pose.p)
        away = torch.atan2(-base_local[:, 0], base_local[:, 1]) + math.pi
        pitch = _TWO_PI / self._wing_count
        best = raw
        best_score = torch.cos(raw - self._rd_approach_lag - away)
        for k in range(1, self._wing_count):
            cand = raw + k * pitch
            score = torch.cos(cand - self._rd_approach_lag - away)
            best = torch.where(score > best_score, cand, best)
            best_score = torch.maximum(score, best_score)
        return best

    def _rd_tcp_bearing(self, tcp_pos: torch.Tensor, near: torch.Tensor) -> torch.Tensor:
        local = self._world_to_fixture(tcp_pos)
        raw = torch.atan2(-local[:, 0], local[:, 1])
        return near + _wrap(raw - near)

    @property
    def _scripted_rider(self) -> Actor:
        return cast(
            "Actor",
            _AimPoint(Pose.create_from_pq(self._rd_arc_point(self._rd_phi, self._rd_contact_local_z()))),
        )

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._rd_commit_radius

    @property
    def _scripted_socket_chamfer(self) -> float:
        return self._rd_commit_ramp

    def _scripted_wait_lower_z(
        self, deck_z: float, rider_pos: torch.Tensor
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        return rider_pos[:, 2] + self._rd_wait_clearance, rider_pos[:, 2]

    def _scripted_hold_gripper(self) -> torch.Tensor:
        _, close_val = self._scripted_gripper_values()
        return torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_rd_phase"):
            self._rd_phase = torch.full((b,), self._PHASE_STROKE, dtype=torch.long, device=device)
            self._rd_phi = torch.zeros((b,), device=device)
            self._rd_bearing = torch.zeros((b,), device=device)
            self._rd_armed = torch.zeros((b,), dtype=torch.bool, device=device)
            self._rd_steps = torch.zeros((b,), dtype=torch.long, device=device)
            self._rd_backoff_phi = torch.zeros((b,), device=device)
            self._rd_stall_steps = torch.zeros((b,), device=device)
            self._rd_last_turn = torch.zeros((b,), device=device)
            self._rd_retries = torch.zeros((b,), dtype=torch.long, device=device)
        bearing = self._rd_pick_wing()
        if env_idx is None:
            self._rd_phase[:] = self._PHASE_STROKE
            self._rd_bearing[:] = bearing
            self._rd_phi[:] = bearing - self._rd_approach_lag
            self._rd_armed[:] = False
            self._rd_steps[:] = 0
            self._rd_stall_steps[:] = 0.0
            self._rd_last_turn[:] = 0.0
            self._rd_retries[:] = 0
        else:
            self._rd_phase[env_idx] = self._PHASE_STROKE
            self._rd_bearing[env_idx] = bearing[env_idx]
            self._rd_phi[env_idx] = bearing[env_idx] - self._rd_approach_lag
            self._rd_armed[env_idx] = False
            self._rd_steps[env_idx] = 0
            self._rd_stall_steps[env_idx] = 0.0
            self._rd_last_turn[env_idx] = 0.0
            self._rd_retries[env_idx] = 0

    def _rd_position_input(self, target_world: torch.Tensor, state: _InterceptState) -> torch.Tensor:
        delta = target_world - state.tcp_pos
        boost = self._rd_work_boost * _smoothstep(self._rd_stall_steps / self._rd_stall_ramp)
        backoff = self._rd_phase == self._PHASE_BACKOFF
        work = torch.where(
            backoff, torch.full_like(state.speed, self._rd_retreat_step), self._rd_work_step + boost
        ).unsqueeze(-1)
        vertical = torch.where(
            backoff,
            torch.full_like(state.speed, self._rd_retreat_step),
            torch.full_like(state.speed, self._rd_vertical_step),
        ).unsqueeze(-1)
        comove = (self._scripted_auto_comove_headroom * state.speed / self.control_freq).unsqueeze(-1)
        work_along, cross_cap = work, work
        if self.velocity_randomization.enabled:
            sde = self._velrand_scale()
            work_along = work * sde[:, 1:2]
            cross_cap = work * sde[:, 0:1]
            vertical = vertical * sde[:, 2:3]
        along_cap = comove + work_along
        along = (delta[:, :2] * state.direction).sum(dim=-1, keepdim=True)
        cross = delta[:, :2] - along * state.direction
        cross = cross * torch.clamp(cross_cap / torch.linalg.norm(cross, dim=-1, keepdim=True).clamp(min=1e-9), max=1.0)
        xy = torch.clamp(along, -along_cap, along_cap) * state.direction + cross
        dz = torch.clamp(delta[:, 2:3], -vertical, vertical)
        delta = torch.cat([xy, dz], dim=-1)
        delta = self._apply_reach_guard(delta, state.tcp_pos)
        self._scripted_prev_delta = delta
        return torch.clip(delta / 0.1, -1.0, 1.0)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_rd_phase"):
            self.reset_scripted()

        state = self._intercept_state()
        self._rd_bearing = self._rd_bearing + self._rd_wrap_wing(self._nut_yaw() - self._rd_bearing)
        bearing = self._rd_bearing
        turn = self._rundown_turn
        target = self._turn_target + self._rd_stroke_margin
        remaining = target - turn

        lower_z = cast("torch.Tensor", state.lower_z)
        oy = self.workspace_offset[1]
        along = (self.fixture.pose.p[:, 1] - oy) * state.direction[:, 1]
        waiting = along < -(self._rd_engage_y + state.speed * self._rd_engage_lead_time)
        self._rd_armed = self._rd_armed | (
            (self._rd_phase == self._PHASE_STROKE)
            & ~waiting
            & (torch.abs(state.tcp_pos[:, 2] - lower_z) < self._rd_arm_z_tol)
            & (state.xy_dist < self._rd_arm_xy_tol)
        )
        stroking = self._rd_armed & (self._rd_phase == self._PHASE_STROKE)

        tcp_bearing = self._rd_tcp_bearing(state.tcp_pos, bearing)
        ramp = (remaining / self._rd_slow_band).clamp(0.0, 1.0)
        gap = self._rd_arc_gap_min + (self._rd_arc_gap - self._rd_arc_gap_min) * ramp
        tcp_radius = torch.linalg.norm(self._world_to_fixture(state.tcp_pos)[:, :2], dim=-1)
        gap = gap * _smoothstep(
            (tcp_radius - self._rd_radius_floor) / (self._wing_contact_radius - self._rd_radius_floor)
        )
        phi_descend = bearing - self._rd_approach_lag
        phi_stroke = torch.minimum(tcp_bearing + gap, bearing + self._rd_wing_lead_max)
        self._rd_phi = torch.where(stroking, torch.maximum(phi_stroke, phi_descend), phi_descend)
        self._rd_steps = torch.where(stroking, self._rd_steps + 1, self._rd_steps)
        arc_world = self._rd_arc_point(self._rd_phi, self._rd_contact_local_z())

        progressing = (turn - self._rd_last_turn) > self._rd_stall_eps
        self._rd_last_turn = turn
        self._rd_stall_steps = torch.where(
            stroking & ~progressing, self._rd_stall_steps + 1.0, torch.zeros_like(self._rd_stall_steps)
        )

        spin = self._nut_spin()
        predicted = turn + spin * self._rd_stop_time
        stalled = (turn >= self._turn_target + self._rd_backoff_floor) & (spin.abs() < self._rd_stall_speed)
        expired = (self._rd_steps > self._rd_stroke_budget) | (self._rd_stall_steps >= self._rd_stall_give_up)
        retry = stroking & expired & (turn < self._turn_target) & (self._rd_retries < self._rd_max_retries)
        to_backoff = stroking & ((predicted >= target) | stalled | (expired & ~retry))
        self._rd_retries = self._rd_retries + retry.long()
        self._rd_armed = self._rd_armed & ~retry
        self._rd_steps = torch.where(retry, torch.zeros_like(self._rd_steps), self._rd_steps)
        self._rd_stall_steps = torch.where(retry, torch.zeros_like(self._rd_stall_steps), self._rd_stall_steps)
        self._rd_last_turn = torch.where(retry, turn, self._rd_last_turn)
        stroking = stroking & ~retry
        self._rd_backoff_phi = torch.where(to_backoff, tcp_bearing, self._rd_backoff_phi)
        self._rd_phase = torch.where(to_backoff, torch.full_like(self._rd_phase, self._PHASE_BACKOFF), self._rd_phase)

        backoff_local = torch.zeros((self.num_envs, 3), device=self.device)
        backoff_local[:, 0] = -self._rd_retract_radius * torch.sin(self._rd_backoff_phi)
        backoff_local[:, 1] = self._rd_retract_radius * torch.cos(self._rd_backoff_phi)
        backoff_local[:, 2] = self._rd_withdraw_z_above
        backoff_target = self._fixture_to_world(backoff_local)
        is_backoff = (self._rd_phase == self._PHASE_BACKOFF).unsqueeze(-1)
        target_world = torch.where(is_backoff, backoff_target, state.target_world)
        target_world[:, 2] = torch.where(stroking, arc_world[:, 2], target_world[:, 2])
        ox, oz = self.workspace_offset[0], self.workspace_offset[2]
        park = torch.empty_like(target_world)
        park[:, 0] = ox
        park[:, 1] = oy - state.direction[:, 1] * self._rd_engage_y
        park[:, 2] = oz + self._conveyor_deck_height + self._rd_wait_z_above
        target_world = torch.where(waiting.unsqueeze(-1), park, target_world)

        pos_input = torch.where(
            self._rd_armed.unsqueeze(-1),
            self._rd_position_input(target_world, state),
            self._position_input(target_world, state.tcp_pos),
        )
        rot_input = self._rotation_input(state.tcp_q)
        gripper = self._scripted_hold_gripper()

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
