from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    rider_lateral_offset: bool = False
    carrier_yaw: bool = False
    hand_start_angle: bool = False
    hub_damping: bool = False

    lateral_range: tuple[float, float] = (-0.015, 0.015)
    yaw_range: tuple[float, float] = (-0.35, 0.35)
    start_angle_range: tuple[float, float] = (-0.14, 0.14)
    hub_damping_range: tuple[float, float] = (0.6, 1.5)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    carrier_yaw: bool = False
    hand_start_angle: bool = False
    hub_damping: bool = False

    yaw_range: tuple[float, float] = (-0.35, 0.35)
    start_angle_range: tuple[float, float] = (-0.14, 0.14)
    hub_damping_range: tuple[float, float] = (0.6, 1.5)

    x_range: tuple[float, float] = (-0.015, 0.015)
    y_range: tuple[float, float] = (-0.20, 0.20)
