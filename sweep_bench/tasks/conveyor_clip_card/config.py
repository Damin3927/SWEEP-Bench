from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    clip_torque: bool = False

    clip_force_range: tuple[float, float] = (0.7, 1.4)

    speed_range: tuple[float, float] = (0.10, 0.40)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    x_range: tuple[float, float] = (-0.007, 0.007)
    y_range: tuple[float, float] = (-0.25, 0.25)

    clip_torque: bool = False
    clip_force_range: tuple[float, float] = (0.7, 1.4)
