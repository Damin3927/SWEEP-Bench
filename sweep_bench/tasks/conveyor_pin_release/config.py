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
    flap_spring: bool = False

    lateral_range: tuple[float, float] = (-0.015, 0.015)
    yaw_range: tuple[float, float] = (-0.12, 0.12)
    spring_range: tuple[float, float] = (0.008, 0.014)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    carrier_yaw: bool = False
    flap_spring: bool = False

    yaw_range: tuple[float, float] = (-0.12, 0.12)
    spring_range: tuple[float, float] = (0.008, 0.014)

    x_range: tuple[float, float] = (-0.015, 0.015)
    y_range: tuple[float, float] = (-0.20, 0.20)
