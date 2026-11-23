from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    rider_lateral_offset: bool = False
    contact_height: bool = False

    lateral_range: tuple[float, float] = (-0.015, 0.015)
    contact_height_range: tuple[float, float] = (0.70, 0.95)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    contact_height: bool = False

    contact_height_range: tuple[float, float] = (0.70, 0.95)

    x_range: tuple[float, float] = (-0.015, 0.015)
    y_range: tuple[float, float] = (-0.25, 0.25)
