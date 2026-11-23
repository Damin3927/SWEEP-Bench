from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    rider_lateral_offset: bool = False
    box_yaw: bool = False
    lid_start_ajar: bool = False

    lateral_range: tuple[float, float] = (-0.015, 0.015)
    yaw_range: tuple[float, float] = (-0.14, 0.14)
    ajar_range: tuple[float, float] = (0.0, 0.14)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    box_yaw: bool = False
    lid_start_ajar: bool = False

    yaw_range: tuple[float, float] = (-0.14, 0.14)
    ajar_range: tuple[float, float] = (0.0, 0.14)

    x_range: tuple[float, float] = (-0.015, 0.015)
    y_range: tuple[float, float] = (-0.20, 0.20)
