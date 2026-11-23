from __future__ import annotations

from dataclasses import dataclass

from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    StaticObjectRandomization,
)


@dataclass
class RandomizationConfig(ConveyorMotionRandomization):
    cart_gap: bool = False
    lead_yaw: bool = False
    socket_clearance: bool = False
    cart_mass: bool = False

    speed_range: tuple[float, float] = (0.10, 0.40)

    cart_gap_range: tuple[float, float] = (0.26, 0.32)
    lead_yaw_range: tuple[float, float] = (-0.045, 0.045)
    channel_half_x_range: tuple[float, float] = (0.020, 0.024)
    lead_density_range: tuple[float, float] = (320.0, 520.0)
    follower_density_range: tuple[float, float] = (480.0, 780.0)


@dataclass
class StaticRandomizationConfig(StaticObjectRandomization):
    cart_gap: bool = False
    lead_yaw: bool = False
    socket_clearance: bool = False
    cart_mass: bool = False

    cart_gap_range: tuple[float, float] = (0.26, 0.32)
    lead_yaw_range: tuple[float, float] = (-0.045, 0.045)
    channel_half_x_range: tuple[float, float] = (0.020, 0.024)
    lead_density_range: tuple[float, float] = (320.0, 520.0)
    follower_density_range: tuple[float, float] = (480.0, 780.0)

    x_range: tuple[float, float] = (-0.02, 0.02)
    y_range: tuple[float, float] = (-0.12, 0.12)
