from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


@dataclass(frozen=True)
class _AimPose:
    p: torch.Tensor


class _MouthAim:
    def __init__(self, env: ScriptedPourFillMixin) -> None:
        self._env = env

    @property
    def pose(self) -> _AimPose:
        env = self._env
        p = env.hopper.pose.p.clone()
        p[:, 1] = p[:, 1] + env._pour_lead * env.belt.direction_sign
        p[:, 2] = p[:, 2] + env._hopper_mouth_rim_local_z
        return _AimPose(p)


class ScriptedPourFillMixin(_ScriptedInsertFamilyMixin):
    hopper: Actor
    _hopper_mouth_rim_local_z: float
    _hopper_well_half: float
    _hopper_mouth_half: float
    _hopper_wall_thickness: float
    _bead_radius: float

    _pour_preroll_angle: float = math.radians(0.0)
    _pour_tip_angle: float = math.radians(55.0)

    _pour_commit_xy: float = 0.045
    _pour_commit_z: float = 0.055
    _pour_commit_fallback_xy: float = 0.10
    _pour_commit_max_wait: int = 12

    _pour_lead: float = 0.044

    _pour_wait_above_rim: float = 0.104
    _pour_z_above_rim: float = 0.070

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._hopper_well_half - self._bead_radius

    @property
    def _scripted_socket_chamfer(self) -> float:
        return self._hopper_mouth_half - (self._hopper_well_half + self._hopper_wall_thickness)

    @property
    def _scripted_rider(self) -> Actor:
        if not hasattr(self, "_pour_mouth_aim"):
            self._pour_mouth_aim = _MouthAim(self)
        return cast("Actor", self._pour_mouth_aim)

    def _scripted_wait_lower_z(
        self, deck_z: float, rider_pos: torch.Tensor
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        return rider_pos[:, 2] + self._pour_wait_above_rim, rider_pos[:, 2] + self._pour_z_above_rim

    def _scripted_hold_gripper(self) -> torch.Tensor:
        _open_val, close_val = self._scripted_gripper_values()
        return torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)

    def _gripper_down_target_q(self) -> torch.Tensor:
        q_down = super()._gripper_down_target_q()
        if not hasattr(self, "_pour_roll_cmd"):
            return q_down
        half = 0.5 * self._pour_roll_cmd
        q_roll = torch.zeros_like(q_down)
        q_roll[:, 0] = torch.cos(half)
        q_roll[:, 2] = torch.sin(half)
        return quaternion_multiply(q_down, q_roll)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_pour_started"):
            self._pour_started = torch.zeros((b,), dtype=torch.bool, device=device)
            self._pour_near_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._pour_roll_cmd = torch.zeros((b,), device=device)
        if env_idx is None:
            self._pour_started[:] = False
            self._pour_near_counter[:] = 0
            self._pour_roll_cmd[:] = 0.0
        else:
            self._pour_started[env_idx] = False
            self._pour_near_counter[env_idx] = 0
            self._pour_roll_cmd[env_idx] = 0.0

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_pour_started"):
            self.reset_scripted()

        state = self._intercept_state()

        at_station = (state.xy_dist < self._pour_commit_xy) & (
            state.tcp_pos[:, 2] < state.lower_z + self._pour_commit_z
        )
        near = state.xy_dist < self._pour_commit_fallback_xy
        self._pour_near_counter = torch.where(
            near, self._pour_near_counter + 1, torch.zeros_like(self._pour_near_counter)
        )
        self._pour_started = self._pour_started | at_station | (self._pour_near_counter >= self._pour_commit_max_wait)
        self._pour_roll_cmd = torch.where(
            self._pour_started,
            torch.full_like(self._pour_roll_cmd, self._pour_tip_angle),
            torch.full_like(self._pour_roll_cmd, self._pour_preroll_angle),
        )

        pos_input = self._position_input(state.target_world, state.tcp_pos)
        rot_input = self._rotation_input(state.tcp_q)
        gripper = self._scripted_hold_gripper()

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
