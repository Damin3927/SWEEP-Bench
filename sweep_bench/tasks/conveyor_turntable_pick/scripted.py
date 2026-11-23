from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedTurntablePickMixin(_ScriptedGraspFamilyMixin):
    part: Actor

    def part_axis_xy(self) -> torch.Tensor:
        raise NotImplementedError

    def disc_spin_command(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_approach_z_above: float = 0.17

    _scripted_xy_align_tol: float = 0.015
    _scripted_close_xy_tol: float = 0.020
    _scripted_close_z_tol: float = 0.012
    _scripted_close_steps: int = 8

    _scripted_yaw_lead_time: float = 0.48

    _grasp_above_centre: float = 0.006

    @property
    def _scripted_rider(self) -> Actor:
        return self.part

    def _scripted_grasp_z(self, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2] + self._grasp_above_centre

    def _led_part_axis_xy(self) -> torch.Tensor:
        u = self.part_axis_xy()
        lead = self.disc_spin_command() * self._scripted_yaw_lead_time
        cos, sin = torch.cos(lead), torch.sin(lead)
        return torch.stack([cos * u[:, 0] - sin * u[:, 1], sin * u[:, 0] + cos * u[:, 1]], dim=-1)

    def _grip_axis_xy(self) -> torch.Tensor:
        if not hasattr(self, "_scripted_grip_axis"):
            return self._led_part_axis_xy()
        return self._scripted_grip_axis

    def _update_grip_axis(self) -> None:
        if not hasattr(self, "_scripted_phase"):
            self.reset_scripted()
        phase = self._scripted_phase
        pre_grasp = (phase == self._PHASE_APPROACH) | (phase == self._PHASE_DESCEND) | (phase == self._PHASE_CLOSE)
        self._scripted_grip_axis = torch.where(
            pre_grasp.unsqueeze(-1), self._led_part_axis_xy(), self._scripted_grip_axis
        )

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        if not hasattr(self, "_scripted_grip_axis"):
            self._scripted_grip_axis = torch.zeros((self.num_envs, 2), device=self.device)
        axis = self._led_part_axis_xy()
        if env_idx is None:
            self._scripted_grip_axis[:] = axis
        else:
            self._scripted_grip_axis[env_idx] = axis[env_idx]

    def compute_scripted_action(self) -> torch.Tensor:
        self._update_grip_axis()
        return super().compute_scripted_action()

    def _gripper_down_target_q(self) -> torch.Tensor:
        device = self.device
        b = self.num_envs

        u = self._grip_axis_xy()
        perp = torch.stack([u[:, 1], -u[:, 0]], dim=-1)

        y_body_world = torch.zeros((b, 3), device=device)
        y_body_world[:, :2] = perp
        z_body_world = torch.zeros((b, 3), device=device)
        z_body_world[:, 2] = -1.0
        x_body_world = torch.cross(y_body_world, z_body_world, dim=-1)

        R = torch.stack([x_body_world, y_body_world, z_body_world], dim=-1)
        return matrix_to_quaternion(R)
