from __future__ import annotations

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


@dataclass(frozen=True)
class _AimPoint:
    pose: Pose


class ScriptedClockHandMixin(_ScriptedInsertFamilyMixin):
    _PHASE_SWEEP = 0
    _PHASE_BACKOFF = 1

    carrier: Actor
    _pad_radius: float
    _contact_local_z: float

    _scripted_hover_z_above: float = 0.20
    _clock_wait_clearance: float = 0.03
    _clock_commit_radius: float = 0.012
    _clock_commit_ramp: float = 0.012
    _clock_arm_xy_tol: float = 0.030
    _clock_arm_z_tol: float = 0.006

    _clock_approach_lag: float = 0.80
    _clock_arc_rate: float = 0.09
    _clock_arc_rate_min: float = 0.025
    _clock_arc_lead: float = 0.30
    _clock_arc_lead_min: float = 0.07
    _clock_slow_band: float = 0.35

    _clock_stop_time: float = 0.05
    _clock_retract_radius: float = 0.10
    _clock_withdraw_z_above: float = 0.12
    _clock_retract_back: float = 0.14
    _clock_sweep_budget: int = 90

    def _clock_hand_angle(self) -> torch.Tensor:
        raise NotImplementedError

    def _clock_hand_spin(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def _clock_target_angle(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def _clock_move_sign(self) -> torch.Tensor:
        raise NotImplementedError

    def _carrier_to_world(self, local: torch.Tensor) -> torch.Tensor:
        rot = quaternion_to_matrix(self.carrier.pose.q)
        return self.carrier.pose.p + torch.bmm(rot, local.unsqueeze(-1)).squeeze(-1)

    def _world_to_carrier(self, world: torch.Tensor) -> torch.Tensor:
        rot = quaternion_to_matrix(self.carrier.pose.q)
        return torch.bmm(rot.transpose(1, 2), (world - self.carrier.pose.p).unsqueeze(-1)).squeeze(-1)

    def _clock_arc_point(self, phi: torch.Tensor, local_z: torch.Tensor | float) -> torch.Tensor:
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 0] = -self._pad_radius * torch.sin(phi)
        local[:, 1] = self._pad_radius * torch.cos(phi)
        local[:, 2] = torch.as_tensor(local_z, device=self.device, dtype=local.dtype).expand(self.num_envs)
        return self._carrier_to_world(local)

    @property
    def _scripted_rider(self) -> Actor:
        return cast(
            "Actor", _AimPoint(Pose.create_from_pq(self._clock_arc_point(self._clock_phi, self._contact_local_z)))
        )

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._clock_commit_radius

    @property
    def _scripted_socket_chamfer(self) -> float:
        return self._clock_commit_ramp

    def _scripted_wait_lower_z(
        self, deck_z: float, rider_pos: torch.Tensor
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        return rider_pos[:, 2] + self._clock_wait_clearance, rider_pos[:, 2]

    def _scripted_hold_gripper(self) -> torch.Tensor:
        _, close_val = self._scripted_gripper_values()
        return torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_clock_phase"):
            self._clock_phase = torch.full((b,), self._PHASE_SWEEP, dtype=torch.long, device=device)
            self._clock_phi = torch.zeros((b,), device=device)
            self._clock_armed = torch.zeros((b,), dtype=torch.bool, device=device)
            self._clock_steps = torch.zeros((b,), dtype=torch.long, device=device)
            self._clock_backoff_phi = torch.zeros((b,), device=device)
        start = self._clock_hand_angle() - self._clock_move_sign * self._clock_approach_lag
        if env_idx is None:
            self._clock_phase[:] = self._PHASE_SWEEP
            self._clock_phi[:] = start
            self._clock_armed[:] = False
            self._clock_steps[:] = 0
        else:
            self._clock_phase[env_idx] = self._PHASE_SWEEP
            self._clock_phi[env_idx] = start[env_idx]
            self._clock_armed[env_idx] = False
            self._clock_steps[env_idx] = 0

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_clock_phase"):
            self.reset_scripted()

        state = self._intercept_state()
        angle = self._clock_hand_angle()
        target = self._clock_target_angle
        sign = self._clock_move_sign
        remaining = sign * (target - angle)

        lower_z = cast("torch.Tensor", state.lower_z)
        self._clock_armed = self._clock_armed | (
            (self._clock_phase == self._PHASE_SWEEP)
            & (torch.abs(state.tcp_pos[:, 2] - lower_z) < self._clock_arm_z_tol)
            & (state.xy_dist < self._clock_arm_xy_tol)
        )
        sweeping = self._clock_armed & (self._clock_phase == self._PHASE_SWEEP)

        ramp = (remaining.abs() / self._clock_slow_band).clamp(0.0, 1.0)
        lead = self._clock_arc_lead_min + (self._clock_arc_lead - self._clock_arc_lead_min) * ramp
        rate = self._clock_arc_rate_min + (self._clock_arc_rate - self._clock_arc_rate_min) * ramp
        edge = angle + sign * lead
        advanced = self._clock_phi + sign * rate
        servo = torch.where(sign > 0, torch.minimum(advanced, edge), torch.maximum(advanced, edge))
        self._clock_phi = torch.where(sweeping, servo, self._clock_phi)
        self._clock_steps = torch.where(sweeping, self._clock_steps + 1, self._clock_steps)
        arc_world = self._clock_arc_point(self._clock_phi, self._contact_local_z)

        predicted = sign * (target - (angle + self._clock_hand_spin() * self._clock_stop_time))
        to_backoff = sweeping & ((predicted <= 0.0) | (self._clock_steps > self._clock_sweep_budget))
        local_tcp = self._world_to_carrier(state.tcp_pos)
        bearing = torch.atan2(-local_tcp[:, 0], local_tcp[:, 1]) - sign * self._clock_retract_back
        self._clock_backoff_phi = torch.where(to_backoff, bearing, self._clock_backoff_phi)
        self._clock_phase = torch.where(
            to_backoff, torch.full_like(self._clock_phase, self._PHASE_BACKOFF), self._clock_phase
        )

        backoff_local = torch.zeros((self.num_envs, 3), device=self.device)
        backoff_local[:, 0] = -self._clock_retract_radius * torch.sin(self._clock_backoff_phi)
        backoff_local[:, 1] = self._clock_retract_radius * torch.cos(self._clock_backoff_phi)
        backoff_local[:, 2] = self._clock_withdraw_z_above
        backoff_target = self._carrier_to_world(backoff_local)
        is_backoff = (self._clock_phase == self._PHASE_BACKOFF).unsqueeze(-1)
        target_world = torch.where(is_backoff, backoff_target, state.target_world)
        target_world[:, 2] = torch.where(sweeping, arc_world[:, 2], target_world[:, 2])

        pos_input = self._position_input(target_world, state.tcp_pos)
        rot_input = self._rotation_input(state.tcp_q)
        gripper = self._scripted_hold_gripper()

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
