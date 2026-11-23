from __future__ import annotations

import math
from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_SPEED_RANGE = (0.10, 0.40)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    speed_range: tuple[float, float] = _SPEED_RANGE

    board_yaw: bool = False
    board_yaw_range: tuple[float, float] = (-math.pi / 12, math.pi / 12)

    clear_path: bool = False
    clear_path_range: tuple[float, float] = (0.8, 1.25)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    board_yaw: bool = False
    board_yaw_range: tuple[float, float] = (-math.pi / 12, math.pi / 12)

    clear_path: bool = False
    clear_path_range: tuple[float, float] = (0.8, 1.25)

    x_range: tuple[float, float] = (-0.0025, 0.0025)
    y_range: tuple[float, float] = (-0.25, 0.25)
