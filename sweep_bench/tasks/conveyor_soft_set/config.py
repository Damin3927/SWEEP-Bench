from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_CARRIER_YAW_RANGE = (-0.209, 0.209)

_CARRIER_X_RANGE = (-0.010, 0.010)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    carrier_yaw: bool = False
    carrier_lateral_offset: bool = False

    carrier_yaw_range: tuple[float, float] = _CARRIER_YAW_RANGE
    carrier_x_range: tuple[float, float] = _CARRIER_X_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    carrier_yaw: bool = False

    carrier_yaw_range: tuple[float, float] = _CARRIER_YAW_RANGE
    x_range: tuple[float, float] = _CARRIER_X_RANGE
    y_range: tuple[float, float] = (-0.20, 0.20)
