from __future__ import annotations

import math
from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_CARRIER_YAW_RANGE = (-math.radians(12.0), math.radians(12.0))
_DETENT_STRENGTH_RANGE = (0.8, 1.15)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    carrier_yaw: bool = False
    detent_strength: bool = False

    carrier_yaw_range: tuple[float, float] = _CARRIER_YAW_RANGE
    detent_strength_range: tuple[float, float] = _DETENT_STRENGTH_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    carrier_yaw: bool = False
    detent_strength: bool = False

    carrier_yaw_range: tuple[float, float] = _CARRIER_YAW_RANGE
    detent_strength_range: tuple[float, float] = _DETENT_STRENGTH_RANGE

    x_range: tuple[float, float] = (-0.015, 0.015)
    y_range: tuple[float, float] = (-0.25, 0.25)
