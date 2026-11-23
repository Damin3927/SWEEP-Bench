from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_OVERHANG_FRAC_RANGE = (0.35, 0.45)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    overhang_side: bool = False
    overhang_frac: bool = False

    overhang_frac_range: tuple[float, float] = _OVERHANG_FRAC_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    overhang_side: bool = False
    overhang_frac: bool = False

    overhang_frac_range: tuple[float, float] = _OVERHANG_FRAC_RANGE

    y_range: tuple[float, float] = (-0.25, 0.25)
