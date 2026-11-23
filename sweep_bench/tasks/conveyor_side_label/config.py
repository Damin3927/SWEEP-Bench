from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    box_lateral: bool = False
    patch_offset: bool = False
    patch_height: bool = False

    speed_range: tuple[float, float] = (0.10, 0.40)

    box_lateral_range: tuple[float, float] = (-0.020, 0.020)

    patch_offset_range: tuple[float, float] = (-0.012, 0.012)
    patch_height_range: tuple[float, float] = (-0.005, 0.005)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    patch_offset: bool = False
    patch_height: bool = False

    x_range: tuple[float, float] = (-0.020, 0.020)
    y_range: tuple[float, float] = (-0.25, 0.25)

    patch_offset_range: tuple[float, float] = (-0.012, 0.012)
    patch_height_range: tuple[float, float] = (-0.005, 0.005)
