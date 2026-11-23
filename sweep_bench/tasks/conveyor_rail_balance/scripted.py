from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedRailBalanceMixin(_ScriptedPlaceFamilyMixin):
    carrier: Actor
    _block_half_z: torch.Tensor
    _rail_height: float

    _scripted_accel_step: float | None = 0.0025

    _scripted_center_r_in: float = 0.015
    _scripted_center_r_out: float = 0.05
    _scripted_lower_xy_tol: float = 0.021
    _scripted_lower_z_tol: float = 0.012
    _scripted_release_steps: int = 6

    _scripted_wait_clearance: float = 0.020
    _scripted_release_rail_gap: float = -0.001
    _scripted_withdraw_rim_margin: float = 0.03

    @property
    def _scripted_rider(self) -> Actor:
        return self.carrier

    def _rail_top_z(self, carrier_pos: torch.Tensor) -> torch.Tensor:
        return carrier_pos[:, 2] + self._rail_height

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return self._rail_top_z(rider_pos) + 2.0 * self._block_half_z

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rail_z = self._rail_top_z(rider_pos)
        wait_z = rail_z + self._scripted_wait_clearance + self._block_half_z
        lower_z = rail_z + self._scripted_release_rail_gap + self._block_half_z
        return wait_z, lower_z

    def _scripted_release_gripper(self) -> torch.Tensor:
        open_val, _ = self._scripted_gripper_values()
        return torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32)
