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
    carrier_yaw_range: tuple[float, float] = (-0.20, 0.20)
    carrier_lateral_range: tuple[float, float] = (-0.020, 0.020)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    x_range: tuple[float, float] = (-0.020, 0.020)
    y_range: tuple[float, float] = (-0.25, 0.25)
