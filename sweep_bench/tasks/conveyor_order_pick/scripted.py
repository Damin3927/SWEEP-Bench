from __future__ import annotations

from typing import TYPE_CHECKING

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedOrderPickMixin(_ScriptedGraspFamilyMixin):
    target: Actor

    _scripted_approach_z_above: float = 0.15

    _scripted_close_xy_tol: float = 0.022

    @property
    def _scripted_rider(self) -> Actor:
        return self.target
