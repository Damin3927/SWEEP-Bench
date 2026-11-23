from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_RACK_YAW_RANGE = (-0.08, 0.08)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    rack_x_jitter: bool = False
    rack_yaw: bool = False

    speed_range: tuple[float, float] = (0.10, 0.40)
    x_jitter_range: tuple[float, float] = (-0.02, 0.02)
    yaw_range: tuple[float, float] = _RACK_YAW_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    rack_yaw: bool = False

    yaw_range: tuple[float, float] = _RACK_YAW_RANGE
    x_range: tuple[float, float] = (-0.02, 0.02)
    y_range: tuple[float, float] = (-0.25, 0.25)
