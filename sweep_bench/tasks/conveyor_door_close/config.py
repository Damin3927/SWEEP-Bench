from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    rider_lateral_offset: bool = False
    cabinet_yaw: bool = False
    door_start_angle: bool = False

    lateral_range: tuple[float, float] = (-0.012, 0.012)
    yaw_range: tuple[float, float] = (-0.10, 0.10)
    start_angle_range: tuple[float, float] = (0.0, 0.44)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    cabinet_yaw: bool = False
    door_start_angle: bool = False

    yaw_range: tuple[float, float] = (-0.10, 0.10)
    start_angle_range: tuple[float, float] = (0.0, 0.44)

    x_range: tuple[float, float] = (-0.012, 0.012)
    y_range: tuple[float, float] = (-0.20, 0.20)
