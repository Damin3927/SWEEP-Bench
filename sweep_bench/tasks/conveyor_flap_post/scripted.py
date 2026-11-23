from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedFlapPostMixin(_ScriptedPlaceFamilyMixin):
    flap_bin: Actor
    _pin_z: float
    _part_half_z: torch.Tensor
    _part_grasp_half: torch.Tensor

    _scripted_accel_step: float | None = 0.0035

    _scripted_hover_z_above: float = 0.28

    _scripted_auto_park_v_lo: float = 0.02
    _scripted_auto_park_v_hi: float = 0.06

    _scripted_center_r_in: float = 0.012
    _scripted_center_r_out: float = 0.07
    _scripted_lower_xy_tol: float = 0.014
    _scripted_lower_z_tol: float = 0.016
    _scripted_release_steps: int = 4
    _scripted_withdraw_rim_margin: float = 0.06

    _scripted_wait_clearance: float = 0.020
    _scripted_plunge_depth: float = 0.038
    _scripted_release_finger_margin: float = 0.002
    _scripted_full_open_clearance: float = 0.02

    @property
    def _scripted_rider(self) -> Actor:
        return self.flap_bin

    def _flap_plane_z(self, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2] + self._pin_z

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return self._flap_plane_z(rider_pos)

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        plane_z = self._flap_plane_z(rider_pos)
        wait_z = plane_z + self._scripted_wait_clearance + self._part_half_z
        lower_z = plane_z - self._scripted_plunge_depth
        return wait_z, lower_z

    def _release_finger_target(self) -> torch.Tensor:
        return torch.clamp(self._part_grasp_half + self._scripted_release_finger_margin, max=0.04)

    def _scripted_release_gripper(self) -> torch.Tensor:
        open_val, _close_val = self._scripted_gripper_values()
        target = self._gripper_action_for_aperture(2.0 * self._release_finger_target()).to(torch.float32)
        tcp_z = cast("Panda", self.agent).tcp.pose.p[:, 2]
        clear = tcp_z > self.flap_bin.pose.p[:, 2] + self._pin_z + self._scripted_full_open_clearance
        return torch.where(clear, torch.full_like(target, open_val), target)
