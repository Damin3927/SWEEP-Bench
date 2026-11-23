from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_MAT_YAW_RANGE = (-0.26, 0.26)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    mat_yaw: bool = False

    mat_yaw_range: tuple[float, float] = _MAT_YAW_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    mat_yaw: bool = False

    mat_yaw_range: tuple[float, float] = _MAT_YAW_RANGE
    x_range: tuple[float, float] = (-0.02, 0.02)
    y_range: tuple[float, float] = (-0.20, 0.20)
