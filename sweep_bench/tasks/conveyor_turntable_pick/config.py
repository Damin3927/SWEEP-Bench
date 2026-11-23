from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_LANE_OFFSET_RANGE = (-0.012, 0.012)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    speed_range: tuple[float, float] = (0.10, 0.40)

    spin_direction: bool = False

    lane_offset: bool = False

    lane_offset_range: tuple[float, float] = _LANE_OFFSET_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    spin_direction: bool = False

    x_range: tuple[float, float] = (-0.012, 0.012)
    y_range: tuple[float, float] = (-0.20, 0.20)
