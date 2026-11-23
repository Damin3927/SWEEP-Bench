from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    speed_range: tuple[float, float] = (0.10, 0.40)

    target_extension: bool = False
    target_extension_range: tuple[float, float] = (0.018, 0.032)

    spring_stiffness: bool = False
    spring_stiffness_range: tuple[float, float] = (18.0, 34.0)

    carrier_yaw: bool = False
    carrier_yaw_range: tuple[float, float] = (-0.1745, 0.1745)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    x_range: tuple[float, float] = (-0.02, 0.02)
    y_range: tuple[float, float] = (-0.25, 0.25)

    target_extension: bool = False
    target_extension_range: tuple[float, float] = (0.018, 0.032)
    spring_stiffness: bool = False
    spring_stiffness_range: tuple[float, float] = (18.0, 34.0)
    carrier_yaw: bool = False
    carrier_yaw_range: tuple[float, float] = (-0.1745, 0.1745)
