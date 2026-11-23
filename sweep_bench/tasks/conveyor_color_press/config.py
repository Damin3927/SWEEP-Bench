from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    panel_yaw: bool = False
    cap_stiffness: bool = False


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    panel_yaw: bool = False
    cap_stiffness: bool = False

    x_range: tuple[float, float] = (-0.005, 0.005)
    y_range: tuple[float, float] = (-0.25, 0.25)
