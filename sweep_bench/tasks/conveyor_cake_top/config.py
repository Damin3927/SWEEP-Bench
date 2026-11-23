from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_CAKE_YAW_RANGE = (-3.141592653589793, 3.141592653589793)

_ZONE_OFFSET_RANGE = (0.028, 0.035)

_ZONE_RADIUS_RANGE = (0.018, 0.030)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    cake_yaw: bool = False
    zone_offset: bool = False
    zone_radius: bool = False

    cake_yaw_range: tuple[float, float] = _CAKE_YAW_RANGE
    zone_offset_range: tuple[float, float] = _ZONE_OFFSET_RANGE
    zone_radius_range: tuple[float, float] = _ZONE_RADIUS_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    cake_yaw: bool = False
    zone_offset: bool = False
    zone_radius: bool = False

    cake_yaw_range: tuple[float, float] = _CAKE_YAW_RANGE
    zone_offset_range: tuple[float, float] = _ZONE_OFFSET_RANGE
    zone_radius_range: tuple[float, float] = _ZONE_RADIUS_RANGE
    x_range: tuple[float, float] = (-0.01, 0.01)
    y_range: tuple[float, float] = (-0.20, 0.20)
