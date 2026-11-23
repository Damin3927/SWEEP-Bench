from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    carrier_yaw: bool = False

    carrier_lateral: bool = False

    speed_range: tuple[float, float] = (0.10, 0.40)
    carrier_yaw_range: tuple[float, float] = (-0.10, 0.10)
    carrier_lateral_range: tuple[float, float] = (-0.03, 0.03)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.25, 0.25)
