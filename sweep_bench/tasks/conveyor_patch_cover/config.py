from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_SPEED_RANGE = (0.10, 0.40)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    named_patch: bool = False
    patch_swap: bool = False

    speed_range: tuple[float, float] = _SPEED_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    named_patch: bool = False
    patch_swap: bool = False

    x_range: tuple[float, float] = (-0.02, 0.02)
    y_range: tuple[float, float] = (-0.16, 0.16)
