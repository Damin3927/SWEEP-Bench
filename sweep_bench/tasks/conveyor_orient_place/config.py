from __future__ import annotations

import math
from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    slot_yaw: bool = False

    yaw_range: tuple[float, float] = (-math.pi / 4, math.pi / 4)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    slot_yaw: bool = False

    yaw_range: tuple[float, float] = (-math.pi / 4, math.pi / 4)

    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.25, 0.25)
