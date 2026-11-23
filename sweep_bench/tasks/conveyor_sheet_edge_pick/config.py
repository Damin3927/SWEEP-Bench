from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    rack_lateral_offset: bool = False
    sheet_seat_offset: bool = False

    lateral_range: tuple[float, float] = (-0.008, 0.008)
    seat_range: tuple[float, float] = (-0.004, 0.004)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    sheet_seat_offset: bool = False

    seat_range: tuple[float, float] = (-0.004, 0.004)

    x_range: tuple[float, float] = (-0.008, 0.008)
    y_range: tuple[float, float] = (-0.20, 0.20)
