from __future__ import annotations

import math
from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    socket_yaw: bool = False

    plate_lateral: bool = False

    yaw_range: tuple[float, float] = (-math.pi, math.pi)

    speed_range: tuple[float, float] = (0.10, 0.40)
    plate_lateral_range: tuple[float, float] = (-0.03, 0.03)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    socket_yaw: bool = False

    yaw_range: tuple[float, float] = (-math.pi, math.pi)

    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.25, 0.25)
