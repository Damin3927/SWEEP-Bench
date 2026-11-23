from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    body_yaw: bool = False
    cap_stiffness: bool = False

    speed_range: tuple[float, float] = (0.10, 0.40)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    body_yaw: bool = False
    cap_stiffness: bool = False

    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.25, 0.25)
