from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    cover_offset: bool = False
    swipe_side: bool = False
    carrier_lateral_offset: bool = False

    lateral_range: tuple[float, float] = (-0.010, 0.010)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    cover_offset: bool = False
    swipe_side: bool = False

    x_range: tuple[float, float] = (-0.010, 0.010)
    y_range: tuple[float, float] = (-0.25, 0.25)
