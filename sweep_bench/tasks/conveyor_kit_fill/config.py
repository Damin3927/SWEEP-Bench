from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_TRAY_YAW_RANGE = (-0.26, 0.26)

_FEEDER_OFFSET_RANGE = (-0.02, 0.02)
_FEEDER_OFFSET_Y_RANGE = (-0.03, 0.03)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    empty_cell_index: bool = False
    tray_yaw: bool = False
    part_in_holder: bool = False
    feeder_position: bool = False

    tray_yaw_range: tuple[float, float] = _TRAY_YAW_RANGE
    feeder_offset_range: tuple[float, float] = _FEEDER_OFFSET_RANGE
    feeder_offset_y_range: tuple[float, float] = _FEEDER_OFFSET_Y_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    empty_cell_index: bool = False
    tray_yaw: bool = False
    part_in_holder: bool = False
    feeder_position: bool = False

    tray_yaw_range: tuple[float, float] = _TRAY_YAW_RANGE
    feeder_offset_range: tuple[float, float] = _FEEDER_OFFSET_RANGE
    feeder_offset_y_range: tuple[float, float] = _FEEDER_OFFSET_Y_RANGE
    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.20, 0.20)
