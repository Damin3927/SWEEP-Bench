from __future__ import annotations

import math
from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_DISPLAY_YAW_RANGE = (-math.pi / 3.0, math.pi / 3.0)

_CARRIER_YAW_RANGE = (-0.21, 0.21)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    reference_yaw: bool = False
    display_position: bool = False
    carrier_yaw: bool = False

    speed_range: tuple[float, float] = (0.10, 0.40)
    display_yaw_range: tuple[float, float] = _DISPLAY_YAW_RANGE
    carrier_yaw_range: tuple[float, float] = _CARRIER_YAW_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    reference_yaw: bool = False
    display_position: bool = False
    carrier_yaw: bool = False

    display_yaw_range: tuple[float, float] = _DISPLAY_YAW_RANGE
    carrier_yaw_range: tuple[float, float] = _CARRIER_YAW_RANGE
    x_range: tuple[float, float] = (-0.02, 0.02)
    y_range: tuple[float, float] = (-0.20, 0.20)
