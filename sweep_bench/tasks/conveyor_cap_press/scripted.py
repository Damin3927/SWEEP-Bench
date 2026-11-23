from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedCapPressMixin(_ScriptedPlaceFamilyMixin):
    jar: Actor
    cap: Actor
    neck_height: float
    _jar_body_half_height: float
    _cap_knob_radius: float
    _cap_grip_to_throat: float
    _cap_seat_depth: float

    _scripted_pos_step: float = 0.015
    _scripted_v_ee_max: float = 0.225

    _scripted_v_align: float = 0.15
    _scripted_max_lead: float = 0.30
    _scripted_cross_scale: float = 0.05
    _scripted_prox_scale: float = 0.05
    _scripted_lead_z_weight: float = 0.0

    _scripted_accel_step: float | None = 0.00375

    _scripted_comove_lead_time: float = 0.0

    _scripted_wait_clearance: float = 0.020
    _scripted_max_descent_step: float = 0.018
    _scripted_seat_overtravel: float = 0.002
    _scripted_press_dwell_steps: int = 8
    _scripted_release_steps: int = 12
    _scripted_release_finger_margin: float = 0.006
    _scripted_hold_margin: float = 0.0035

    _scripted_withdraw_rim_margin: float = 0.04

    _scripted_lower_z_tol: float = 0.005
    _scripted_lower_xy_tol: float = 0.008

    _scripted_center_r_in: float = 0.006
    _scripted_center_r_out: float = 0.012

    @property
    def _scripted_rider(self) -> Actor:
        return self.jar

    def _clamp_pos_delta(self, delta_pos: torch.Tensor) -> torch.Tensor:
        clamped = super()._clamp_pos_delta(delta_pos)
        limited = torch.cat(
            [clamped[:, :2], clamped[:, 2:3].clamp(min=-self._scripted_max_descent_step)],
            dim=-1,
        )
        self._scripted_prev_delta = limited
        return limited

    def _neck_top_z(self, deck_z: float) -> float:
        return deck_z + 2.0 * self._jar_body_half_height + self.neck_height

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> float:
        return self._neck_top_z(deck_z)

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, float]:
        neck_top_z = self._neck_top_z(deck_z)
        wait_z = neck_top_z + self._scripted_wait_clearance + self._cap_grip_to_throat
        lower_z = neck_top_z - self._cap_seat_depth - self._scripted_seat_overtravel + self._cap_grip_to_throat
        return wait_z, lower_z

    def _scripted_gripper_values(self) -> tuple[float, float]:
        open_val, _close_val = super()._scripted_gripper_values()
        hold_target = self._cap_knob_radius - self._scripted_hold_margin
        return open_val, self._gripper_action_for_aperture(2.0 * hold_target)

    def _scripted_release_gripper(self) -> torch.Tensor:
        open_target = min(self._cap_knob_radius + self._scripted_release_finger_margin, 0.04)
        open_val = self._gripper_action_for_aperture(2.0 * open_target)
        _, hold_val = self._scripted_gripper_values()
        dwelling = (self._scripted_phase == self._PHASE_RELEASE) & (
            self._scripted_release_counter <= self._scripted_press_dwell_steps
        )
        return torch.where(
            dwelling,
            torch.full((self.num_envs,), hold_val, device=self.device, dtype=torch.float32),
            torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32),
        )
