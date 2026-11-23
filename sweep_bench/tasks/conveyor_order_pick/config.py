from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    ordinal: bool = False

    slot_pitch: bool = False

    block_yaw: bool = False

    prompt_paraphrase: bool = False

    pitch_range: tuple[float, float] = (0.10, 0.12)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    ordinal: bool = False
    slot_pitch: bool = False
    block_yaw: bool = False
    prompt_paraphrase: bool = False

    pitch_range: tuple[float, float] = (0.10, 0.12)

    x_range: tuple[float, float] = (-0.03, 0.03)
    y_range: tuple[float, float] = (-0.20, 0.20)
