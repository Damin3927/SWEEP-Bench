from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    apple_orientation: bool = False


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    apple_orientation: bool = False
