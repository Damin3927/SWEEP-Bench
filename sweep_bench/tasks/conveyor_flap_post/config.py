from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_BIN_YAW_RANGE = (-0.21, 0.21)
_TORQUE_SCALE_RANGE = (0.6, 1.6)
_AJAR_RANGE = (0.20, 0.40)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    bin_yaw: bool = False
    flap_return_torque: bool = False
    flap_start_ajar: bool = False

    bin_yaw_range: tuple[float, float] = _BIN_YAW_RANGE
    torque_scale_range: tuple[float, float] = _TORQUE_SCALE_RANGE
    ajar_range: tuple[float, float] = _AJAR_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    bin_yaw: bool = False
    flap_return_torque: bool = False
    flap_start_ajar: bool = False

    bin_yaw_range: tuple[float, float] = _BIN_YAW_RANGE
    torque_scale_range: tuple[float, float] = _TORQUE_SCALE_RANGE
    ajar_range: tuple[float, float] = _AJAR_RANGE
    x_range: tuple[float, float] = (-0.015, 0.015)
    y_range: tuple[float, float] = (-0.20, 0.20)
