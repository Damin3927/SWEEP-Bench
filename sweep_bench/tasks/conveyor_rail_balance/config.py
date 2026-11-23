from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_CARRIER_X_RANGE = (-0.020, 0.020)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    carrier_lateral_offset: bool = False

    carrier_x_range: tuple[float, float] = _CARRIER_X_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    x_range: tuple[float, float] = _CARRIER_X_RANGE
    y_range: tuple[float, float] = (-0.20, 0.20)
