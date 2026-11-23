from __future__ import annotations

import math
from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_BELT_B_SPEED_RANGE = (0.10, 0.50)
_PLACE_Y_RANGE = (-0.30, 0.30)
_RIDER_YAW_RANGE = (-math.pi / 4, math.pi / 4)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    belt_b_speed: bool = False
    place_y: bool = False
    rider_yaw: bool = False
    distractor: bool = False

    belt_b_speed_range: tuple[float, float] = _BELT_B_SPEED_RANGE
    place_y_range: tuple[float, float] = _PLACE_Y_RANGE
    rider_yaw_range: tuple[float, float] = _RIDER_YAW_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    place_y: bool = False
    rider_yaw: bool = False
    distractor: bool = False

    place_y_range: tuple[float, float] = _PLACE_Y_RANGE
    rider_yaw_range: tuple[float, float] = _RIDER_YAW_RANGE

    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.20, 0.20)
