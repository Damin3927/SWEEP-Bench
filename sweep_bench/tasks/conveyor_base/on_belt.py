from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

if TYPE_CHECKING:
    from mani_skill.utils.structs.pose import Pose

DEFAULT_TILT_COS: float = math.cos(math.radians(10.0))
DEFAULT_COMOVE_TOL_FRAC: float = 0.40
DEFAULT_COMOVE_TOL_FLOOR: float = 0.065
DEFAULT_COMOVE_TOL_CEILING: float = 0.220

DEFAULT_MAX_RISE: float = 0.005
FIXED_TOL_SPEED_ENV = "SWEEP_FIXED_TOL_SPEED"


def fixed_tol_speed() -> float | None:
    value = os.environ.get(FIXED_TOL_SPEED_ENV)
    return float(value) if value else None


def tolerance_speed(belt_velocity: torch.Tensor) -> torch.Tensor:
    speed = torch.abs(belt_velocity)
    fixed = fixed_tol_speed()
    if fixed is None:
        return speed
    return torch.where(speed > 0.0, torch.full_like(speed, fixed), speed)


def carried_speed(belt_speed: torch.Tensor, frac: float) -> torch.Tensor:
    fixed = fixed_tol_speed()
    if fixed is None:
        return frac * belt_speed
    return torch.where(belt_speed > 0.0, belt_speed - (1.0 - frac) * fixed, frac * belt_speed)


def belt_carries(velocity_y: torch.Tensor, belt_velocity: torch.Tensor, frac: float) -> torch.Tensor:
    if fixed_tol_speed() is None:
        return velocity_y * belt_velocity >= frac * belt_velocity**2
    speed = torch.abs(belt_velocity)
    return velocity_y * belt_velocity >= carried_speed(speed, frac) * speed


def belt_comove_tol(
    belt_velocity: torch.Tensor,
    *,
    frac: float = DEFAULT_COMOVE_TOL_FRAC,
    floor: float = DEFAULT_COMOVE_TOL_FLOOR,
    ceiling: float = DEFAULT_COMOVE_TOL_CEILING,
) -> torch.Tensor:
    return torch.clamp(frac * tolerance_speed(belt_velocity), min=floor, max=ceiling)


def rider_is_level(pose: Pose, *, tilt_cos: float = DEFAULT_TILT_COS) -> torch.Tensor:
    return quaternion_to_matrix(pose.q)[:, 2, 2] > tilt_cos


def rider_comoves(
    linear_velocity: torch.Tensor,
    belt_velocity: torch.Tensor,
    *,
    frac: float = DEFAULT_COMOVE_TOL_FRAC,
    floor: float = DEFAULT_COMOVE_TOL_FLOOR,
    ceiling: float = DEFAULT_COMOVE_TOL_CEILING,
) -> torch.Tensor:
    tol = belt_comove_tol(belt_velocity, frac=frac, floor=floor, ceiling=ceiling)
    return torch.abs(linear_velocity[:, 1] - belt_velocity) <= tol


def rider_at_rest_height(
    pose: Pose,
    *,
    deck_z: float,
    rest_height: float,
    max_rise: float = DEFAULT_MAX_RISE,
) -> torch.Tensor:
    return (pose.p[:, 2] - (deck_z + rest_height)).abs() <= max_rise


def rider_on_belt(
    *,
    pose: Pose,
    linear_velocity: torch.Tensor,
    belt_velocity: torch.Tensor,
    deck_z: float,
    centre_x: float,
    centre_y: float,
    x_tol: float,
    y_tol: float,
    rest_height: float,
    max_rise: float = DEFAULT_MAX_RISE,
    tilt_cos: float = DEFAULT_TILT_COS,
    comove_tol_frac: float = DEFAULT_COMOVE_TOL_FRAC,
    comove_tol_floor: float = DEFAULT_COMOVE_TOL_FLOOR,
    comove_tol_ceiling: float = DEFAULT_COMOVE_TOL_CEILING,
) -> torch.Tensor:
    pos = pose.p
    return (
        rider_at_rest_height(pose, deck_z=deck_z, rest_height=rest_height, max_rise=max_rise)
        & (torch.abs(pos[:, 0] - centre_x) <= x_tol)
        & (torch.abs(pos[:, 1] - centre_y) <= y_tol)
        & rider_is_level(pose, tilt_cos=tilt_cos)
        & rider_comoves(
            linear_velocity,
            belt_velocity,
            frac=comove_tol_frac,
            floor=comove_tol_floor,
            ceiling=comove_tol_ceiling,
        )
    )
