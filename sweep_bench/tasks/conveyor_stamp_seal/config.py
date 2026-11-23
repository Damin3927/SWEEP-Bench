from __future__ import annotations

import math
from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    pad_yaw: bool = False
    pad_yaw_range: tuple[float, float] = (-math.pi / 4, math.pi / 4)

    detent_stiffness: bool = False
    detent_stiffness_range: tuple[float, float] = (0.7, 1.4)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    pad_yaw: bool = False
    pad_yaw_range: tuple[float, float] = (-math.pi / 4, math.pi / 4)

    detent_stiffness: bool = False
    detent_stiffness_range: tuple[float, float] = (0.7, 1.4)

    x_range: tuple[float, float] = (-0.02, 0.02)
    y_range: tuple[float, float] = (-0.25, 0.25)
