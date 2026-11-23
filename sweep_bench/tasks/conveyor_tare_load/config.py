from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    spring_bin: bool = False
    plunger_side: bool = False
    scale_side: bool = False


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    spring_bin: bool = False
    plunger_side: bool = False
    scale_side: bool = False

    x_range: tuple[float, float] = (-0.02, 0.02)
    y_range: tuple[float, float] = (-0.20, 0.20)
