from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_HOPPER_X_RANGE = (-0.018, 0.018)

_SPEED_RANGE = (0.10, 0.40)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    speed_range: tuple[float, float] = _SPEED_RANGE

    hopper_lateral: bool = False
    hopper_x_range: tuple[float, float] = _HOPPER_X_RANGE

    hopper_yaw: bool = False
    hopper_yaw_range: tuple[float, float] = (-0.35, 0.35)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    hopper_yaw: bool = False
    hopper_yaw_range: tuple[float, float] = (-0.35, 0.35)

    x_range: tuple[float, float] = _HOPPER_X_RANGE
