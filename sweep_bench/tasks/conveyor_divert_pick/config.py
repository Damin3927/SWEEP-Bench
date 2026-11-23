from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_LANE_PHASE_RANGE = (-0.12, 0.0)

_STATIC_ALONG_RAIL_RANGE = (-0.09, 0.09)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    speed_range: tuple[float, float] = (0.10, 0.40)

    lane_phase: bool = False

    lane_phase_range: tuple[float, float] = _LANE_PHASE_RANGE


@dataclass
class StaticDivertPickConfig(StaticObjectRandomization):
    x_range: tuple[float, float] = (0.0, 0.0)
    y_range: tuple[float, float] = _STATIC_ALONG_RAIL_RANGE
