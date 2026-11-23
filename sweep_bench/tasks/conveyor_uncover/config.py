from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    bin_yaw: bool = False

    lid_offset: bool = False

    speed_range: tuple[float, float] = (0.10, 0.40)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    bin_yaw: bool = False
    lid_offset: bool = False

    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.25, 0.25)
