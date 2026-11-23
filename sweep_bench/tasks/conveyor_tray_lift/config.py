from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    payload_offset: bool = False

    payload_offset_frac: float = 0.6


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    payload_offset: bool = False
    payload_offset_frac: float = 0.6
