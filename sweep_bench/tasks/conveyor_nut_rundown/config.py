from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_SPEED_RANGE = (0.10, 0.40)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    rider_lateral_offset: bool = False
    fixture_yaw: bool = False
    nut_start_angle: bool = False
    nut_start_height: bool = False

    speed_range: tuple[float, float] = _SPEED_RANGE

    lateral_range: tuple[float, float] = (-0.015, 0.015)
    yaw_range: tuple[float, float] = (-0.25, 0.25)
    start_angle_range: tuple[float, float] = (-1.571, 1.571)
    start_height_range: tuple[float, float] = (0.0028, 0.0045)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    fixture_yaw: bool = False
    nut_start_angle: bool = False
    nut_start_height: bool = False

    yaw_range: tuple[float, float] = (-0.25, 0.25)
    start_angle_range: tuple[float, float] = (-1.571, 1.571)
    start_height_range: tuple[float, float] = (0.0028, 0.0045)

    x_range: tuple[float, float] = (-0.015, 0.015)
    y_range: tuple[float, float] = (-0.20, 0.20)
