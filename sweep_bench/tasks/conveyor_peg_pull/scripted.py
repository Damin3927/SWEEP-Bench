from __future__ import annotations

from typing import TYPE_CHECKING

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedPullMixin(_ScriptedGraspFamilyMixin):
    peg: Actor
    _peg_radius: float

    _scripted_hold_margin: float = 0.0015

    _scripted_approach_z_above: float = 0.13

    _scripted_auto_bias_v_lo: float = 0.0
    _scripted_auto_bias_v_hi: float = 0.02

    @property
    def _scripted_rider(self) -> Actor:
        return self.peg

    def _scripted_gripper_values(self) -> tuple[float, float]:
        open_val, _full_close = super()._scripted_gripper_values()
        target = self._peg_radius - self._scripted_hold_margin
        close_val = self._gripper_action_for_aperture(2.0 * target)
        return open_val, close_val
