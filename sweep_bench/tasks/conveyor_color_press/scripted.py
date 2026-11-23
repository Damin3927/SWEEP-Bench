from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedColorPressMixin(_ScriptedInsertFamilyMixin):
    caps: list[Actor]
    _panel_origin_above_deck: float
    _cap_half_height: float
    _cap_half_width: float
    _cap_rest_gap: float

    _scripted_accel_step: float | None = 0.0025

    _scripted_wait_clearance: float = 0.02

    _scripted_finger_reach: float = 0.0095

    _scripted_press_overshoot: float = 0.0195
    _scripted_press_min_overshoot: float = 0.001
    _scripted_press_ramp_v_lo: float = 0.10
    _scripted_press_ramp_v_hi: float = 0.50

    _scripted_commit_inset: float = 0.001

    @property
    def _scripted_rider(self) -> Actor:
        return self.caps[0]

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._cap_half_width - self._scripted_commit_inset

    @property
    def _scripted_socket_chamfer(self) -> float:
        return 0.002

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, torch.Tensor]:
        cap_pressed_top = deck_z + self._panel_origin_above_deck + 2.0 * self._cap_half_height
        cap_rest_top = cap_pressed_top + self._cap_rest_gap
        wait_z = cap_rest_top + self._scripted_finger_reach + self._scripted_wait_clearance

        speed = torch.abs(self.belt.velocity)
        ramp = (
            (speed - self._scripted_press_ramp_v_lo) / (self._scripted_press_ramp_v_hi - self._scripted_press_ramp_v_lo)
        ).clamp(0.0, 1.0)
        overshoot = self._scripted_press_min_overshoot + ramp * (
            self._scripted_press_overshoot - self._scripted_press_min_overshoot
        )
        return wait_z, cap_pressed_top + self._scripted_finger_reach - overshoot

    def _scripted_hold_gripper(self) -> torch.Tensor:
        _, close_val = self._scripted_gripper_values()
        return torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)
