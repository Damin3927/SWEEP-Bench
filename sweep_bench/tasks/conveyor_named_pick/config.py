from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    target_slot: bool = False

    rider_spacing: bool = False

    rider_yaw: bool = False

    spacing_range: tuple[float, float] = (0.18, 0.22)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    target_slot: bool = False
    rider_spacing: bool = False
    rider_yaw: bool = False
    spacing_range: tuple[float, float] = (0.18, 0.22)
