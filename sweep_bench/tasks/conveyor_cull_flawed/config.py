from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    speed_range: tuple[float, float] = (0.10, 0.40)

    flaw_rate: bool = False
    flaw_rate_range: tuple[float, float] = (0.25, 0.75)

    bin_side: bool = False
    bin_jitter: bool = False
    bin_jitter_x_range: tuple[float, float] = (-0.03, 0.03)
    bin_jitter_y_range: tuple[float, float] = (-0.03, 0.03)

    band_appearance: bool = False


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    flaw_rate: bool = False
    flaw_rate_range: tuple[float, float] = (0.25, 0.75)

    bin_side: bool = False
    bin_jitter: bool = False
    bin_jitter_x_range: tuple[float, float] = (-0.03, 0.03)
    bin_jitter_y_range: tuple[float, float] = (-0.03, 0.03)

    band_appearance: bool = False

    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.20, 0.20)
