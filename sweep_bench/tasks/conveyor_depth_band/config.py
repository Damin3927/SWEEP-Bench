from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    housing_yaw: bool = False
    spring_k: bool = False
    band_offset: bool = False

    spring_k_range: tuple[float, float] = (100.0, 200.0)
    band_offset_range: tuple[float, float] = (-0.0015, 0.0015)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    housing_yaw: bool = False
    spring_k: bool = False
    band_offset: bool = False

    spring_k_range: tuple[float, float] = (100.0, 200.0)
    band_offset_range: tuple[float, float] = (-0.0015, 0.0015)

    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.25, 0.25)
