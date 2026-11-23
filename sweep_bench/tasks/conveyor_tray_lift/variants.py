from __future__ import annotations

import functools

from sweep_bench.assets.catalog import ObjectSpec
from sweep_bench.tasks.conveyor_tray_lift.assets import build_tray

TRAY_WALL_THICKNESS = 0.005
TRAY_WALL_HALF_HEIGHT = 0.015
TRAY_FLOOR_HALF_HEIGHT = 0.004

TRAY_INNER_HALF_X = 0.022

_INNER_HALF_Y = (0.032, 0.026, 0.040)
_COLORS: dict[str, tuple[float, float, float, float]] = {
    "gray": (0.55, 0.57, 0.60, 1.0),
    "blue": (0.13, 0.30, 0.75, 1.0),
    "green": (0.13, 0.60, 0.20, 1.0),
    "red": (0.85, 0.12, 0.12, 1.0),
}

_DENSITY = 1000.0


def _tray_volume(inner_half_x: float, inner_half_y: float) -> float:
    wt, whh, fhh = TRAY_WALL_THICKNESS, TRAY_WALL_HALF_HEIGHT, TRAY_FLOOR_HALF_HEIGHT
    outer_half_x, outer_half_y = inner_half_x + 2 * wt, inner_half_y + 2 * wt
    floor = 8 * outer_half_x * outer_half_y * fhh
    x_walls = 2 * (8 * wt * outer_half_y * whh)
    y_walls = 2 * (8 * inner_half_x * wt * whh)
    return floor + x_walls + y_walls


def _mm(x: float) -> str:
    return f"{round(x * 1000):03d}"


def _tray_spec(inner_half_y: float, color_name: str) -> ObjectSpec:
    inner_half_x = TRAY_INNER_HALF_X
    outer_half_x = inner_half_x + 2 * TRAY_WALL_THICKNESS
    outer_half_y = inner_half_y + 2 * TRAY_WALL_THICKNESS
    half_z = TRAY_FLOOR_HALF_HEIGHT + TRAY_WALL_HALF_HEIGHT
    return ObjectSpec(
        id=f"proc_tray_{_mm(2 * outer_half_x)}x{_mm(2 * outer_half_y)}_{color_name}",
        role="container",
        source="procedural",
        builder_fn=functools.partial(
            build_tray,
            inner_half_x=inner_half_x,
            inner_half_y=inner_half_y,
            wall_thickness=TRAY_WALL_THICKNESS,
            wall_half_height=TRAY_WALL_HALF_HEIGHT,
            floor_half_height=TRAY_FLOOR_HALF_HEIGHT,
            color=_COLORS[color_name],
        ),
        bbox=(outer_half_x, outer_half_y, half_z),
        mass=_DENSITY * _tray_volume(inner_half_x, inner_half_y),
        patch_radius=min(outer_half_x, outer_half_y),
        grasp_width=2 * outer_half_x,
        tags=("procedural", "tray", "container", color_name),
    )


TRAY_LIFT_TRAY_SPECS: tuple[ObjectSpec, ...] = tuple(
    _tray_spec(inner_half_y, color_name) for inner_half_y in _INNER_HALF_Y for color_name in _COLORS
)

TRAY_LIFT_OBJECT_IDS: tuple[str, ...] = tuple(spec.id for spec in TRAY_LIFT_TRAY_SPECS)
