from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedCrushLimitPickMixin(_ScriptedGraspFamilyMixin):
    core: Actor

    _scripted_grip_aperture: float = 0.0540

    _scripted_approach_z_above: float = 0.09

    _scripted_close_lead_time: float = 0.10

    _scripted_lift_scale_cap: float = 3.0

    _scripted_touch_aperture: float = 0.062
    _scripted_squeeze_ramp_steps: int = 8

    @property
    def _scripted_rider(self) -> Actor:
        return self.core

    def compute_scripted_action(self) -> torch.Tensor:
        action = super().compute_scripted_action()
        in_close = self._scripted_phase == self._PHASE_CLOSE
        frac = (self._scripted_close_counter.float() / max(self._scripted_squeeze_ramp_steps, 1)).clamp(0.0, 1.0)
        aperture = self._scripted_touch_aperture + frac * (self._scripted_grip_aperture - self._scripted_touch_aperture)
        ramped = self._gripper_action_for_aperture(aperture).to(action.dtype)
        action[:, 6] = torch.where(in_close, ramped, action[:, 6])
        return action

    def _update_auto_speed_scale(self, tcp_pos: torch.Tensor, rider_pos: torch.Tensor) -> None:
        super()._update_auto_speed_scale(tcp_pos, rider_pos)
        scale = self._scripted_auto_scale
        if scale is None or not hasattr(self, "_scripted_phase"):
            return
        lifting = (self._scripted_phase == self._PHASE_LIFT) | (self._scripted_phase == self._PHASE_HOLD)
        self._scripted_auto_scale = torch.where(lifting, scale.clamp(max=self._scripted_lift_scale_cap), scale)

    def _scripted_gripper_values(self) -> tuple[float, float]:
        open_val, _close_val = super()._scripted_gripper_values()
        return open_val, self._gripper_action_for_aperture(self._scripted_grip_aperture)
