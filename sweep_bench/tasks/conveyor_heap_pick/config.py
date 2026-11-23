from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_CORRAL_LATERAL_RANGE = (-0.010, 0.010)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    riser_slot: bool = False
    corral_lateral: bool = False

    lateral_range: tuple[float, float] = _CORRAL_LATERAL_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    riser_slot: bool = False

    x_range: tuple[float, float] = _CORRAL_LATERAL_RANGE
    y_range: tuple[float, float] = (-0.20, 0.20)
