from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_PALLET_YAW_RANGE = (-0.05, 0.05)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    pallet_yaw: bool = False

    pallet_yaw_range: tuple[float, float] = _PALLET_YAW_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    pallet_yaw: bool = False

    pallet_yaw_range: tuple[float, float] = _PALLET_YAW_RANGE
    x_range: tuple[float, float] = (-0.02, 0.02)
    y_range: tuple[float, float] = (0.0, 0.20)
