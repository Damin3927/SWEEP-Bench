from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedEvenLoadMixin(_ScriptedPlaceFamilyMixin):
    carrier: Actor
    _bar_half_thickness: float
    _pad_top_above_housing: float

    _scripted_accel_step: float | None = 0.0025

    _scripted_comove_lead_time: float = 0.13

    _scripted_center_r_in: float = 0.020
    _scripted_center_r_out: float = 0.055
    _scripted_lower_xy_tol: float = 0.012
    _scripted_lower_z_tol: float = 0.012
    _scripted_release_steps: int = 20
    _scripted_release_hold_steps: int = 16
    _scripted_withdraw_rim_margin: float = 0.05

    _scripted_wait_clearance: float = 0.035
    _scripted_release_pad_gap: float = 0.004

    @property
    def _scripted_rider(self) -> Actor:
        return self.carrier

    def _pad_top_z(self, carrier_pos: torch.Tensor) -> torch.Tensor:
        return carrier_pos[:, 2] + self._pad_top_above_housing

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return self._pad_top_z(rider_pos)

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad_z = self._pad_top_z(rider_pos)
        wait_z = pad_z + self._scripted_wait_clearance + self._bar_half_thickness
        lower_z = pad_z + self._scripted_release_pad_gap + self._bar_half_thickness
        return wait_z, lower_z

    def _scripted_release_gripper(self) -> torch.Tensor:
        open_val, close_val = self._scripted_gripper_values()
        held = (self._scripted_phase == self._PHASE_RELEASE) & (
            self._scripted_release_counter <= self._scripted_release_hold_steps
        )
        return torch.where(
            held,
            torch.full((self.num_envs,), close_val, device=self.device, dtype=torch.float32),
            torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32),
        )
