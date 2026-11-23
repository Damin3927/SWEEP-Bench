from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    shell_lateral_offset: bool = False
    shell_yaw: bool = False

    lateral_range: tuple[float, float] = (-0.030, 0.030)
    yaw_range: tuple[float, float] = (-0.105, 0.105)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    shell_yaw: bool = False

    yaw_range: tuple[float, float] = (-0.105, 0.105)

    x_range: tuple[float, float] = (-0.030, 0.030)
    y_range: tuple[float, float] = (-0.20, 0.20)
