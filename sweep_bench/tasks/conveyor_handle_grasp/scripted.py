from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


@dataclass(frozen=True)
class _AimPose:
    p: torch.Tensor


class _HandleBarAim:
    def __init__(self, env: ScriptedHandleGraspMixin) -> None:
        self._env = env

    @property
    def pose(self) -> _AimPose:
        return _AimPose(self._env.handle_grip_world_pos())

    @property
    def linear_velocity(self) -> torch.Tensor:
        return self._env.handle_vessel_linear_velocity()


class ScriptedHandleGraspMixin(_ScriptedGraspFamilyMixin):
    def handle_grip_world_pos(self) -> torch.Tensor:
        raise NotImplementedError

    def handle_axis_xy(self) -> torch.Tensor:
        raise NotImplementedError

    def handle_vessel_linear_velocity(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_close_xy_tol: float = 0.018
    _scripted_close_z_tol: float = 0.018

    _scripted_approach_z_above: float = 0.17

    @property
    def _scripted_rider(self) -> Actor:
        if not hasattr(self, "_scripted_bar_aim"):
            self._scripted_bar_aim = _HandleBarAim(self)
        return cast("Actor", self._scripted_bar_aim)

    def _gripper_down_target_q(self) -> torch.Tensor:
        device = self.device
        b = self.num_envs

        d = self._scripted_motion_dir
        default_perp = torch.stack([d[:, 1], -d[:, 0]], dim=-1)

        u = self.handle_axis_xy()
        perp = torch.stack([u[:, 1], -u[:, 0]], dim=-1)
        flip = torch.where((perp * default_perp).sum(dim=-1, keepdim=True) >= 0.0, 1.0, -1.0)
        perp = perp * flip

        y_body_world = torch.zeros((b, 3), device=device)
        y_body_world[:, :2] = perp
        z_body_world = torch.zeros((b, 3), device=device)
        z_body_world[:, 2] = -1.0
        x_body_world = torch.cross(y_body_world, z_body_world, dim=-1)

        R = torch.stack([x_body_world, y_body_world, z_body_world], dim=-1)
        return matrix_to_quaternion(R)
