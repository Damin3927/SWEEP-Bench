from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    rider_lateral_offset: bool = False
    carrier_yaw: bool = False

    lateral_range: tuple[float, float] = (-0.012, 0.012)
    yaw_range: tuple[float, float] = (-0.10, 0.10)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    carrier_yaw: bool = False

    yaw_range: tuple[float, float] = (-0.10, 0.10)

    x_range: tuple[float, float] = (-0.012, 0.012)
    y_range: tuple[float, float] = (-0.20, 0.20)
