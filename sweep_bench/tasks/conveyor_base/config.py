from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from sweep_bench.tasks.conveyor_base.randomization_scene import SceneRandomizationConfig

SceneStyle = Literal["table", "replicacad", "ai2thor", "robocasa"]
PlatformStyle = Literal["box", "table"]
ConveyorDirection = Literal["+y", "-y"]


@dataclass
class ConveyorMotionRandomization(SceneRandomizationConfig):
    speed: bool = False
    direction: bool = False

    speed_range: tuple[float, float] = (0.10, 0.40)
    directions: tuple[ConveyorDirection, ...] = ("+y", "-y")

    spawn_phase_range: float = 0.10

    accel: float = 0.0
    accel_window: float = 1.0
    accel_range: tuple[float, float] = (0.0, 0.0)


@dataclass
class StaticObjectRandomization(SceneRandomizationConfig):
    object_position: bool = False

    x_range: tuple[float, float] = (-0.04, 0.04)
    y_range: tuple[float, float] = (-0.30, 0.30)


AnyRandomization = ConveyorMotionRandomization | StaticObjectRandomization


@dataclass
class VelocityRandomizationConfig:
    enabled: bool = False
    theta: float = 2.0
    alpha: float = 0.6
    target: float = 0.9
    min_scale: float = 0.7
    max_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.target < 1.0:
            raise ValueError(
                f"velocity randomization target must be strictly inside (0, 1), got {self.target}. "
                f"Both endpoints absorb the process (drift and diffusion vanish together there), "
                f"which silently disables the velocity axis."
            )
        if self.theta <= 0.0 or self.alpha <= 0.0:
            raise ValueError(f"theta and alpha must be positive, got theta={self.theta}, alpha={self.alpha}")
        for name in ("min_scale", "max_scale", "target", "theta", "alpha"):
            value = getattr(self, name, None)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"{name} must be finite, got {value}. A non-finite bound passes the ordering "
                    "checks below -- NaN fails every comparison and infinity is honestly ordered -- "
                    "and then maps the latent to a non-finite action scale that nothing downstream "
                    "can detect."
                )
        if self.min_scale <= 0.0:
            raise ValueError(
                f"min_scale must be positive, got {self.min_scale}. A floor at or below zero lets a draw "
                f"stop the arm outright or reverse its step, which is not a pace perturbation."
            )
        if self.min_scale > self.max_scale:
            raise ValueError(
                f"min_scale must not exceed max_scale, got min_scale={self.min_scale}, "
                f"max_scale={self.max_scale}. Inverted bounds map the latent backwards -- a latent at its "
                f"ceiling would produce the SLOWEST step -- and nothing downstream can detect it."
            )
