from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion, quaternion_multiply

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@dataclass(frozen=True)
class _AimPose:
    p: torch.Tensor


class _MatAim:
    def __init__(self, env: ScriptedShoeMatMixin) -> None:
        self._env = env

    @property
    def pose(self) -> _AimPose:
        return _AimPose(self._env.release_aim_point())


class ScriptedShoeMatMixin(_ScriptedPlaceFamilyMixin):
    _mat_half_z: float
    mat: Actor

    def held_roll_correction_q(self) -> torch.Tensor:
        raise NotImplementedError

    def held_drop_offset(self) -> torch.Tensor:
        raise NotImplementedError

    def sole_down_cos(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_accel_step: float | None = 0.0025

    _scripted_center_r_in: float = 0.025
    _scripted_center_r_out: float = 0.06
    _scripted_lower_xy_tol: float = 0.025
    _scripted_lower_z_tol: float = 0.007
    _scripted_release_steps: int = 6

    _scripted_wait_clearance: float = 0.030
    _scripted_release_gap: float = 0.015

    _scripted_roll_lead_cos: float = 0.985

    _scripted_descent_brake_frac: float = 0.5
    _scripted_min_descent_step: float = 0.003
    _scripted_max_descent_step: float = 0.02

    _scripted_release_tilt_cos: float = 0.985
    _scripted_blocked_r_in: float = 0.0005
    _scripted_blocked_r_out: float = 0.001
    _scripted_stopped_blocked_r_in: float = -0.002
    _scripted_stopped_blocked_r_out: float = -0.001

    _scripted_release_comove_lead: float = 0.10

    def release_aim_point(self) -> torch.Tensor:
        p = self.mat.pose.p.clone()
        releasing = self._scripted_phase == self._PHASE_RELEASE
        led = p[:, 1] + self.belt.velocity * self._scripted_release_comove_lead
        p[:, 1] = torch.where(releasing, led, p[:, 1])
        return p

    @property
    def _scripted_rider(self) -> Actor:
        if not hasattr(self, "_scripted_mat_aim"):
            self._scripted_mat_aim = _MatAim(self)
        return cast("Actor", self._scripted_mat_aim)

    def _gripper_down_target_q(self) -> torch.Tensor:
        agent = cast("Panda", self.agent)
        b, device = self.num_envs, self.device
        d = self._scripted_motion_dir

        y_body_world = torch.zeros((b, 3), device=device)
        y_body_world[:, :2] = d
        z_body_world = torch.zeros((b, 3), device=device)
        z_body_world[:, 2] = -1.0
        x_body_world = torch.cross(y_body_world, z_body_world, dim=-1)
        along = matrix_to_quaternion(torch.stack([x_body_world, y_body_world, z_body_world], dim=-1))
        flipped = quaternion_multiply(along, self._flip_quat())
        nearer = (
            torch.abs((agent.tcp.pose.q * along).sum(dim=-1)) >= torch.abs((agent.tcp.pose.q * flipped).sum(dim=-1))
        ).unsqueeze(-1)
        return quaternion_multiply(torch.where(nearer, along, flipped), self.held_roll_correction_q())

    def _flip_quat(self) -> torch.Tensor:
        q = torch.zeros((self.num_envs, 4), device=self.device)
        q[:, 3] = 1.0
        return q

    def _scripted_commit_window(self) -> tuple[torch.Tensor, torch.Tensor]:
        flat = self.sole_down_cos() >= self._scripted_release_tilt_cos
        stopped = self._scripted_belt_stopped()

        def per_env(open_value: float, blocked_value: float, stopped_value: float) -> torch.Tensor:
            blocked = torch.full((self.num_envs,), blocked_value, device=self.device, dtype=torch.float32)
            blocked = blocked.masked_fill(stopped, stopped_value)
            return blocked.masked_fill(flat, open_value)

        return (
            per_env(self._scripted_center_r_in, self._scripted_blocked_r_in, self._scripted_stopped_blocked_r_in),
            per_env(self._scripted_center_r_out, self._scripted_blocked_r_out, self._scripted_stopped_blocked_r_out),
        )

    def _roll_lead_gain(self) -> torch.Tensor:
        agent = cast("Panda", self.agent)
        rot_cos = torch.abs((agent.tcp.pose.q * self._gripper_down_target_q()).sum(dim=-1))
        t = ((rot_cos - self._scripted_roll_lead_cos) / (1.0 - self._scripted_roll_lead_cos)).clamp(0.0, 1.0)
        return (t * t * (3.0 - 2.0 * t)).unsqueeze(-1)

    def _clamp_pos_delta(self, delta_pos: torch.Tensor) -> torch.Tensor:
        agent = cast("Panda", self.agent)
        limited = delta_pos * self._roll_lead_gain()
        room = (agent.tcp.pose.p[:, 2] - self._scripted_release_z()).clamp(min=0.0)
        cap = (self._scripted_descent_brake_frac * room).clamp(
            min=self._scripted_min_descent_step, max=self._scripted_max_descent_step
        )
        limited[:, 2] = torch.maximum(limited[:, 2], -cap)
        return super()._clamp_pos_delta(limited)

    def _scripted_mat_top_z(self) -> torch.Tensor:
        return self.mat.pose.p[:, 2] + self._mat_half_z

    def _scripted_sweep_z(self) -> torch.Tensor:
        return self._scripted_mat_top_z() + self._scripted_wait_clearance + self.held_drop_offset()

    def _scripted_release_z(self) -> torch.Tensor:
        return self._scripted_mat_top_z() + self._scripted_release_gap + self.held_drop_offset()

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2] + self._mat_half_z

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._scripted_sweep_z(), self._scripted_release_z()

    def _scripted_release_gripper(self) -> torch.Tensor:
        open_val, _ = self._scripted_gripper_values()
        return torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32)
