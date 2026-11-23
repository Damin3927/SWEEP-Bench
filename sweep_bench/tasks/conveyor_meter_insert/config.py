from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    target_depth: bool = False
    speed_range: tuple[float, float] = (0.10, 0.40)
    target_depth_range: tuple[float, float] = (0.020, 0.038)

    bore_yaw: bool = False
    bore_yaw_range: tuple[float, float] = (-0.7854, 0.7854)

    bore_clearance: bool = False
    bore_clear_radius_range: tuple[float, float] = (0.004, 0.008)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.25, 0.25)

    target_depth: bool = False
    target_depth_range: tuple[float, float] = (0.020, 0.038)
    bore_yaw: bool = False
    bore_yaw_range: tuple[float, float] = (-0.7854, 0.7854)
    bore_clearance: bool = False
    bore_clear_radius_range: tuple[float, float] = (0.004, 0.008)
