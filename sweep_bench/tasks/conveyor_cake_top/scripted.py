from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


@dataclass(frozen=True)
class _AimPose:
    p: torch.Tensor


class _TargetZoneAim:
    def __init__(self, env: ScriptedCakeTopMixin) -> None:
        self._env = env

    @property
    def pose(self) -> _AimPose:
        return _AimPose(self._env.zone_world_pos())


class ScriptedCakeTopMixin(_ScriptedPlaceFamilyMixin):
    _topping_half_z: torch.Tensor

    def zone_world_pos(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_accel_step: float | None = 0.0025

    _scripted_center_r_in: float = 0.010
    _scripted_center_r_out: float = 0.05
    _scripted_lower_xy_tol: float = 0.004
    _scripted_release_steps: int = 5

    _scripted_wait_clearance: float = 0.040
    _scripted_release_gap: float = 0.003

    @property
    def _scripted_rider(self) -> Actor:
        if not hasattr(self, "_scripted_zone_aim"):
            self._scripted_zone_aim = _TargetZoneAim(self)
        return cast("Actor", self._scripted_zone_aim)

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2]

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        crown_z = rider_pos[:, 2]
        wait_z = crown_z + self._scripted_wait_clearance + self._topping_half_z
        lower_z = crown_z + self._scripted_release_gap + self._topping_half_z
        return wait_z, lower_z

    def _scripted_release_gripper(self) -> torch.Tensor:
        open_val, _ = self._scripted_gripper_values()
        return torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32)
