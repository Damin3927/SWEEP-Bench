from __future__ import annotations

from typing import TYPE_CHECKING

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedGraspMixin(_ScriptedGraspFamilyMixin):
    apple: Actor

    _scripted_approach_z_above: float = 0.17

    @property
    def _scripted_rider(self) -> Actor:
        return self.apple
