from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_BELT_B_SPEED_RANGE = (0.02, 0.30)
_PLACE_X_RANGE = (-0.30, -0.18)
_RIDER_YAW_RANGE = (-0.26, 0.26)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    belt_b_speed: bool = False
    place_x: bool = False
    rider_yaw: bool = False
    distractor: bool = False
    branch_side: bool = False

    belt_b_speed_range: tuple[float, float] = _BELT_B_SPEED_RANGE
    place_x_range: tuple[float, float] = _PLACE_X_RANGE
    rider_yaw_range: tuple[float, float] = _RIDER_YAW_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    place_x: bool = False
    rider_yaw: bool = False
    distractor: bool = False
    branch_side: bool = False

    place_x_range: tuple[float, float] = _PLACE_X_RANGE
    rider_yaw_range: tuple[float, float] = _RIDER_YAW_RANGE

    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.20, 0.20)
