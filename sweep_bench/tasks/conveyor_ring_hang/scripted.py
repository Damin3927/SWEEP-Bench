from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedPlaceMixin(_ScriptedPlaceFamilyMixin):
    stand: Actor
    _base_half_height: float
    _post_height: float
    _ring_half_height: float
    _ring_half_width: float
    _aperture_half_width: float
    _post_radius: float

    _scripted_accel_step: float | None = 0.0025

    _scripted_wait_clearance: float = 0.015
    _scripted_engage: float = 0.015
    _scripted_release_finger_margin: float = 0.008

    _scripted_center_r_in: float = 0.012
    _scripted_center_r_out: float = 0.045
    _scripted_lower_xy_tol: float = 0.015
    _scripted_lower_z_tol: float = 0.012

    @property
    def _scripted_rider(self) -> Actor:
        return self.stand

    @property
    def _post_top_above_deck(self) -> float:
        return 2.0 * self._base_half_height + self._post_height

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> float:
        return deck_z + self._post_top_above_deck

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, float]:
        post_top = deck_z + self._post_top_above_deck
        wait_z = post_top + self._scripted_wait_clearance + self._ring_half_height
        lower_z = post_top - self._scripted_engage + self._ring_half_height
        return wait_z, lower_z

    def _release_finger_target(self) -> float:
        return min(self._ring_half_width + self._scripted_release_finger_margin, 0.04)

    def _scripted_release_gripper(self) -> torch.Tensor:
        value = self._gripper_action_for_aperture(2.0 * self._release_finger_target())
        return torch.full((self.num_envs,), value, device=self.device, dtype=torch.float32)
