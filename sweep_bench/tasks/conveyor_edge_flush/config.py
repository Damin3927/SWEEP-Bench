from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_SKEW_OFFSET_RANGE = (0.020, 0.050)
_SKEW_YAW_RANGE_DEG = (-30.0, 30.0)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    rail_side: bool = False

    initial_skew: bool = False
    skew_offset_range: tuple[float, float] = _SKEW_OFFSET_RANGE
    skew_yaw_range_deg: tuple[float, float] = _SKEW_YAW_RANGE_DEG


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    rail_side: bool = False

    initial_skew: bool = False
    skew_offset_range: tuple[float, float] = _SKEW_OFFSET_RANGE
    skew_yaw_range_deg: tuple[float, float] = _SKEW_YAW_RANGE_DEG

    x_range: tuple[float, float] = (-0.01, 0.01)
    y_range: tuple[float, float] = (-0.25, 0.25)
