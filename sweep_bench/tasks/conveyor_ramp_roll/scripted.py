from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedRampRollMixin(_ScriptedPlaceFamilyMixin):
    carrier: Actor
    _ball_seat_local_z: torch.Tensor
    _ball_radius: torch.Tensor
    _ramp_top_local_z: float
    _channel_wall_height: float

    _scripted_center_r_in: float = 0.015
    _scripted_center_r_out: float = 0.05
    _scripted_lower_xy_tol: float = 0.012
    _scripted_lower_z_tol: float = 0.012
    _scripted_release_steps: int = 4
    _scripted_withdraw_rim_margin: float = 0.08

    _scripted_wait_clearance: float = 0.030
    _scripted_release_gap: float = 0.004

    _scripted_hold_finger_target: float = 0.014
    _scripted_release_finger_margin: float = 0.008

    @property
    def _scripted_rider(self) -> Actor:
        return self.carrier

    def _scripted_gripper_values(self) -> tuple[float, float]:
        open_val, _close_val = super()._scripted_gripper_values()
        return open_val, self._gripper_action_for_aperture(2.0 * self._scripted_hold_finger_target)

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2] + (self._ramp_top_local_z + self._channel_wall_height)

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lower_z = rider_pos[:, 2] + self._ball_seat_local_z + self._scripted_release_gap
        return lower_z + self._scripted_wait_clearance, lower_z

    def _release_finger_target(self) -> torch.Tensor:
        return torch.clamp(self._ball_radius + self._scripted_release_finger_margin, max=0.04)

    def _scripted_release_gripper(self) -> torch.Tensor:
        return self._gripper_action_for_aperture(2.0 * self._release_finger_target()).to(torch.float32)
