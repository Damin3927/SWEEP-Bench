from __future__ import annotations

import math
from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_TRAY_START_OUT_RANGE = (0.020, 0.030)
_SLEEVE_YAW_RANGE = (-math.radians(15.0), math.radians(15.0))


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    tray_start_out: bool = False
    sleeve_yaw: bool = False

    tray_start_out_range: tuple[float, float] = _TRAY_START_OUT_RANGE
    sleeve_yaw_range: tuple[float, float] = _SLEEVE_YAW_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    tray_start_out: bool = False
    sleeve_yaw: bool = False

    tray_start_out_range: tuple[float, float] = _TRAY_START_OUT_RANGE
    sleeve_yaw_range: tuple[float, float] = _SLEEVE_YAW_RANGE

    x_range: tuple[float, float] = (-0.02, 0.02)
    y_range: tuple[float, float] = (-0.25, 0.25)
