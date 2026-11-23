from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedPlaceMixin(_ScriptedPlaceFamilyMixin):
    bowl: Actor
    _bowl_half_height: float
    _bowl_interior_floor_offset: float
    _fruit_half_height: torch.Tensor
    _fruit_half_width: torch.Tensor

    _scripted_accel_step: float | None = 0.0025

    _scripted_wait_clearance: float = 0.015
    _scripted_lower_gap: float = 0.012
    _scripted_release_finger_margin: float = 0.012

    @property
    def _scripted_rider(self) -> Actor:
        return self.bowl

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> float:
        return deck_z + 2.0 * self._bowl_half_height

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._scripted_wait_z(deck_z), self._scripted_lower_z(rider_pos[:, 2])

    def _scripted_wait_z(self, deck_z: float) -> torch.Tensor:
        rim_z = deck_z + 2.0 * self._bowl_half_height
        return rim_z + self._fruit_half_height + self._scripted_wait_clearance

    def _scripted_lower_z(self, bowl_z: torch.Tensor) -> torch.Tensor:
        rest_z = bowl_z + self._bowl_interior_floor_offset + self._fruit_half_height
        soft_z = rest_z + self._scripted_lower_gap
        aperture_z = bowl_z + 2.0 * self._release_finger_target() - 0.052
        return torch.maximum(soft_z, aperture_z)

    def _release_finger_target(self) -> torch.Tensor:
        return torch.clamp(self._fruit_half_width + self._scripted_release_finger_margin, max=0.04)

    def _scripted_release_gripper(self) -> torch.Tensor:
        return self._gripper_action_for_aperture(2.0 * self._release_finger_target()).to(torch.float32)
