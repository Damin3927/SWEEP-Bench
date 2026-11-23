from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_PATCH_YAW_RANGE = (-50.0, 50.0)
_PATCH_HEIGHT_RANGE = (-0.010, 0.010)
_STANDOFF_TARGET_RANGE = (0.050, 0.070)
_BLOCK_LATERAL_RANGE = (-0.015, 0.015)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    patch_height: bool = False

    standoff_target: bool = False

    block_lateral: bool = False

    speed_range: tuple[float, float] = (0.10, 0.40)

    patch_yaw_range: tuple[float, float] = _PATCH_YAW_RANGE
    patch_height_range: tuple[float, float] = _PATCH_HEIGHT_RANGE
    standoff_target_range: tuple[float, float] = _STANDOFF_TARGET_RANGE
    block_lateral_range: tuple[float, float] = _BLOCK_LATERAL_RANGE


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    patch_height: bool = False
    standoff_target: bool = False

    x_range: tuple[float, float] = _BLOCK_LATERAL_RANGE
    y_range: tuple[float, float] = (-0.25, 0.25)

    patch_yaw_range: tuple[float, float] = _PATCH_YAW_RANGE
    patch_height_range: tuple[float, float] = _PATCH_HEIGHT_RANGE
    standoff_target_range: tuple[float, float] = _STANDOFF_TARGET_RANGE
