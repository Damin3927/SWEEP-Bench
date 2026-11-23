from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    bead_interference: bool = False
    jar_lateral: bool = False
    cap_color: bool = False

    jar_lateral_range: tuple[float, float] = (-0.025, 0.025)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    bead_interference: bool = False
    cap_color: bool = False

    x_range: tuple[float, float] = (-0.025, 0.025)
    y_range: tuple[float, float] = (-0.25, 0.25)
