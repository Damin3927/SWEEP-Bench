from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_CARRIER_YAW_RANGE = (-0.14, 0.14)

_PART_OFFSET_RANGE = (-0.015, 0.015)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    carrier_yaw: bool = False
    part_offset: bool = False

    carrier_yaw_range: tuple[float, float] = _CARRIER_YAW_RANGE
    part_offset_range: tuple[float, float] = _PART_OFFSET_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    carrier_yaw: bool = False
    part_offset: bool = False

    carrier_yaw_range: tuple[float, float] = _CARRIER_YAW_RANGE
    part_offset_range: tuple[float, float] = _PART_OFFSET_RANGE
    x_range: tuple[float, float] = (-0.02, 0.02)
    y_range: tuple[float, float] = (-0.20, 0.20)
