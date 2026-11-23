from __future__ import annotations

import math
from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)

HANDLE_YAW_DEFAULT: float = math.radians(30.0)
HANDLE_YAW_WIDE: float = math.radians(60.0)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    handle_yaw: bool = False


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    handle_yaw: bool = False
