from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_CARRIER_YAW_RANGE = (-0.10, 0.10)

_SPEED_RANGE = (0.10, 0.40)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    carrier_yaw: bool = False

    speed_range: tuple[float, float] = _SPEED_RANGE
    carrier_yaw_range: tuple[float, float] = _CARRIER_YAW_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    carrier_yaw: bool = False

    carrier_yaw_range: tuple[float, float] = _CARRIER_YAW_RANGE
    x_range: tuple[float, float] = (-0.02, 0.02)
    y_range: tuple[float, float] = (-0.20, 0.20)
