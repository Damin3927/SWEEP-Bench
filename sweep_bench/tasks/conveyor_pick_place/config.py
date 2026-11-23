from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    bin_position: bool = False

    bin_x_range: tuple[float, float] = (-0.32, -0.24)
    bin_side_y_range: tuple[float, float] = (0.28, 0.42)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    bin_position: bool = False

    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.22, 0.22)

    bin_x_range: tuple[float, float] = (-0.32, -0.24)
    bin_side_y_range: tuple[float, float] = (0.28, 0.42)
