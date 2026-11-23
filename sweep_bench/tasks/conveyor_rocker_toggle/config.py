from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    toggle_start_side: bool = False
    housing_yaw: bool = False
    rocker_stiffness: bool = False

    housing_yaw_range: tuple[float, float] = (-0.35, 0.35)
    rocker_stiffness_range: tuple[float, float] = (0.6, 1.6)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    toggle_start_side: bool = False
    housing_yaw: bool = False
    rocker_stiffness: bool = False

    housing_yaw_range: tuple[float, float] = (-0.35, 0.35)
    rocker_stiffness_range: tuple[float, float] = (0.6, 1.6)

    x_range: tuple[float, float] = (-0.025, 0.025)
    y_range: tuple[float, float] = (-0.25, 0.25)
