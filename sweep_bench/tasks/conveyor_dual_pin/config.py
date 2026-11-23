from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    pin_yaw: bool = False

    base_lateral: bool = False

    speed_range: tuple[float, float] = (0.10, 0.40)

    yaw_range: tuple[float, float] = (-0.20, 0.20)

    base_lateral_range: tuple[float, float] = (-0.03, 0.03)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    pin_yaw: bool = False

    yaw_range: tuple[float, float] = (-0.20, 0.20)

    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.25, 0.25)
