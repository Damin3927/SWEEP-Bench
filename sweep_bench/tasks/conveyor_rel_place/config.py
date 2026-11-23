from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    relation: bool = False


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    relation: bool = False

    x_range: tuple[float, float] = (-0.01, 0.01)
    y_range: tuple[float, float] = (-0.16, 0.16)
