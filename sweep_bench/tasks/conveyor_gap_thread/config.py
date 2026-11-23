from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_CRATE_YAW_RANGE = (-0.175, 0.175)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    crate_yaw: bool = False

    crate_yaw_range: tuple[float, float] = _CRATE_YAW_RANGE
    speed_range: tuple[float, float] = (0.10, 0.40)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    crate_yaw: bool = False

    crate_yaw_range: tuple[float, float] = _CRATE_YAW_RANGE
    x_range: tuple[float, float] = (-0.015, 0.015)
    y_range: tuple[float, float] = (-0.20, 0.20)
