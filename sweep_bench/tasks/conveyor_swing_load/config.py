from __future__ import annotations

import math
from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    speed_range: tuple[float, float] = (0.10, 0.40)

    pendant_start_swing: bool = False

    swing_range: tuple[float, float] = (-math.radians(8.0), math.radians(8.0))


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    pendant_start_swing: bool = False
    swing_range: tuple[float, float] = (-math.radians(8.0), math.radians(8.0))
