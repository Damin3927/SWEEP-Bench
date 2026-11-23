from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    bin_side: bool = False
    bin_x_range: tuple[float, float] = (-0.30, -0.26)
    bin_side_y_range: tuple[float, float] = (0.32, 0.38)

    payload_offset: bool = False
    payload_offset_frac: float = 0.6


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    bin_side: bool = False
    bin_x_range: tuple[float, float] = (-0.30, -0.26)
    bin_side_y_range: tuple[float, float] = (0.32, 0.38)
    payload_offset: bool = False
    payload_offset_frac: float = 0.6
