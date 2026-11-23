from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_CARRIER_PITCH_RANGE = (0.20, 0.30)

_TRAY_YAW_RANGE = (-0.10, 0.10)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    carrier_pitch: bool = False
    tray_yaw: bool = False

    carrier_pitch_range: tuple[float, float] = _CARRIER_PITCH_RANGE
    tray_yaw_range: tuple[float, float] = _TRAY_YAW_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    carrier_pitch: bool = False
    tray_yaw: bool = False

    carrier_pitch_range: tuple[float, float] = _CARRIER_PITCH_RANGE
    tray_yaw_range: tuple[float, float] = _TRAY_YAW_RANGE
    x_range: tuple[float, float] = (-0.008, 0.008)
    y_range: tuple[float, float] = (-0.15, 0.15)
