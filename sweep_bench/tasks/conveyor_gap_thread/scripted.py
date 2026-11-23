from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedGapThreadMixin(_ScriptedPlaceFamilyMixin):
    _rider_half_z: torch.Tensor
    _release_floor_gap: torch.Tensor
    _crate_floor_local_z: float
    _divider_top_local_z: float
    crate: Actor

    _scripted_accel_step: float | None = None

    _scripted_center_r_in: float = 0.008
    _scripted_center_r_out: float = 0.05
    _scripted_lower_xy_tol: float = 0.008
    _scripted_lower_z_tol: float = 0.010
    _scripted_release_steps: int = 5

    _scripted_wait_clearance: float = 0.015

    @property
    def _scripted_rider(self) -> Actor:
        return self.crate

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2] + self._divider_top_local_z

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wait_z = rider_pos[:, 2] + self._divider_top_local_z + self._scripted_wait_clearance + self._rider_half_z
        lower_z = rider_pos[:, 2] + self._crate_floor_local_z + self._release_floor_gap + self._rider_half_z
        return wait_z, lower_z

    def _scripted_release_gripper(self) -> torch.Tensor:
        open_val, _ = self._scripted_gripper_values()
        return torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32)
