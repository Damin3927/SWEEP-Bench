from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    rider_lateral_offset: bool = False
    sleeve_yaw: bool = False
    tray_start_out: bool = False

    lateral_range: tuple[float, float] = (-0.015, 0.015)
    yaw_range: tuple[float, float] = (-0.175, 0.175)
    start_out_range: tuple[float, float] = (0.0, 0.2)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    sleeve_yaw: bool = False
    tray_start_out: bool = False

    yaw_range: tuple[float, float] = (-0.175, 0.175)
    start_out_range: tuple[float, float] = (0.0, 0.2)

    x_range: tuple[float, float] = (-0.015, 0.015)
    y_range: tuple[float, float] = (-0.25, 0.25)
