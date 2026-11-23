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


class _NamedPatchAim:
    def __init__(self, env: ScriptedPatchCoverMixin) -> None:
        self._env = env

    @property
    def pose(self) -> _AimPose:
        return _AimPose(self._env.named_patch_world_pos())


class ScriptedPatchCoverMixin(_ScriptedPlaceFamilyMixin):
    _tile_half_z: torch.Tensor

    def named_patch_world_pos(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_accel_step: float | None = 0.0025

    _scripted_comove_lead_time: float = 0.10

    _scripted_lower_xy_tol: float = 0.015
    _scripted_lower_z_tol: float = 0.012
    _scripted_center_r_in: float = 0.015
    _scripted_center_r_out: float = 0.055
    _scripted_release_steps: int = 10
    _scripted_withdraw_rim_margin: float = 0.03

    _scripted_wait_clearance: float = 0.050
    _scripted_release_gap: float = 0.004

    @property
    def _scripted_rider(self) -> Actor:
        if not hasattr(self, "_scripted_patch_aim"):
            self._scripted_patch_aim = _NamedPatchAim(self)
        return cast("Actor", self._scripted_patch_aim)

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2] + 2.0 * self._tile_half_z

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        top_z = rider_pos[:, 2]
        lower_z = top_z + self._scripted_release_gap + self._tile_half_z
        wait_z = lower_z + self._scripted_wait_clearance
        return wait_z, lower_z

    def _scripted_release_gripper(self) -> torch.Tensor:
        open_val, _ = self._scripted_gripper_values()
        return torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32)
