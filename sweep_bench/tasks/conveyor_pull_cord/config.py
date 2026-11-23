from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    rider_lateral_offset: bool = False
    rider_yaw: bool = False
    spring_stiffness: bool = False

    speed_range: tuple[float, float] = (0.10, 0.40)
    lateral_range: tuple[float, float] = (-0.02, 0.02)
    yaw_range: tuple[float, float] = (-0.06, 0.06)
    spring_k_range: tuple[float, float] = (0.7, 1.3)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    rider_yaw: bool = False
    spring_stiffness: bool = False

    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.25, 0.25)
    yaw_range: tuple[float, float] = (-0.06, 0.06)
    spring_k_range: tuple[float, float] = (0.7, 1.3)
