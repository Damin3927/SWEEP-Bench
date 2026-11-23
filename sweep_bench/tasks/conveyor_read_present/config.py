from __future__ import annotations

import math
from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

_MARK_JITTER = math.radians(20.0)

_READER_XY_JITTER = 0.03
_READER_Z_JITTER = 0.025


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    mark_face: bool = False
    reader_pose: bool = False

    mark_face_range: tuple[float, float] = (-_MARK_JITTER, _MARK_JITTER)
    reader_dx_range: tuple[float, float] = (-_READER_XY_JITTER, _READER_XY_JITTER)
    reader_dy_range: tuple[float, float] = (-_READER_XY_JITTER, _READER_XY_JITTER)
    reader_dz_range: tuple[float, float] = (-_READER_Z_JITTER, _READER_Z_JITTER)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    mark_face: bool = False
    reader_pose: bool = False

    mark_face_range: tuple[float, float] = (-_MARK_JITTER, _MARK_JITTER)
    reader_dx_range: tuple[float, float] = (-_READER_XY_JITTER, _READER_XY_JITTER)
    reader_dy_range: tuple[float, float] = (-_READER_XY_JITTER, _READER_XY_JITTER)
    reader_dz_range: tuple[float, float] = (-_READER_Z_JITTER, _READER_Z_JITTER)

    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.22, 0.22)
