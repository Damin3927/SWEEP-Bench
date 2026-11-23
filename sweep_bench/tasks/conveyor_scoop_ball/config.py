from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    ball_lateral_offset: bool = False
    capture_lead: bool = False
    scoop_width: bool = False

    speed_range: tuple[float, float] = (0.10, 0.40)

    ball_x_range: tuple[float, float] = (-0.015, 0.015)
    capture_lead_range: tuple[float, float] = (0.08, 0.12)
    scoop_inner_half_x_range: tuple[float, float] = (0.038, 0.050)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    capture_lead: bool = False
    scoop_width: bool = False

    capture_lead_range: tuple[float, float] = (0.08, 0.12)
    scoop_inner_half_x_range: tuple[float, float] = (0.038, 0.050)

    x_range: tuple[float, float] = (-0.015, 0.015)
    y_range: tuple[float, float] = (-0.20, 0.20)
