from __future__ import annotations

import functools

from sweep_bench.assets.catalog import ObjectSpec
from sweep_bench.tasks.conveyor_decant.assets import build_cup_shell

CUP_WALL_THICKNESS = 0.005
CUP_WALL_HALF_HEIGHT = 0.020
CUP_FLOOR_HALF_HEIGHT = 0.004

CUP_INNER_HALF_X = 0.022


_INNER_HALF_Y = (0.026, 0.030)
_COLORS: dict[str, tuple[float, float, float, float]] = {
    "stone": (0.80, 0.78, 0.72, 1.0),
    "blue": (0.13, 0.30, 0.75, 1.0),
    "green": (0.13, 0.60, 0.20, 1.0),
}

CUP_DENSITY = 400.0


def _cup_volume(inner_half_x: float, inner_half_y: float) -> float:
    wt, whh, fhh = CUP_WALL_THICKNESS, CUP_WALL_HALF_HEIGHT, CUP_FLOOR_HALF_HEIGHT
    outer_half_x, outer_half_y = inner_half_x + 2 * wt, inner_half_y + 2 * wt
    floor = 8 * outer_half_x * outer_half_y * fhh
    x_walls = 2 * (8 * wt * outer_half_y * whh)
    y_walls = 2 * (8 * inner_half_x * wt * whh)
    return floor + x_walls + y_walls


def _mm(x: float) -> str:
    return f"{round(x * 1000):03d}"


def _cup_spec(inner_half_y: float, color_name: str) -> ObjectSpec:
    inner_half_x = CUP_INNER_HALF_X
    outer_half_x = inner_half_x + 2 * CUP_WALL_THICKNESS
    outer_half_y = inner_half_y + 2 * CUP_WALL_THICKNESS
    half_z = CUP_FLOOR_HALF_HEIGHT + CUP_WALL_HALF_HEIGHT
    return ObjectSpec(
        id=f"proc_cup_{_mm(2 * outer_half_x)}x{_mm(2 * outer_half_y)}_{color_name}",
        role="container",
        source="procedural",
        builder_fn=functools.partial(
            build_cup_shell,
            inner_half_x=inner_half_x,
            inner_half_y=inner_half_y,
            wall_thickness=CUP_WALL_THICKNESS,
            wall_half_height=CUP_WALL_HALF_HEIGHT,
            floor_half_height=CUP_FLOOR_HALF_HEIGHT,
            color=_COLORS[color_name],
            density=CUP_DENSITY,
        ),
        bbox=(outer_half_x, outer_half_y, half_z),
        mass=CUP_DENSITY * _cup_volume(inner_half_x, inner_half_y),
        patch_radius=min(outer_half_x, outer_half_y),
        grasp_width=2 * outer_half_x,
        tags=("procedural", "cup", "container", color_name),
    )


DECANT_CUP_SPECS: tuple[ObjectSpec, ...] = tuple(
    _cup_spec(inner_half_y, color_name) for inner_half_y in _INNER_HALF_Y for color_name in _COLORS
)

DECANT_OBJECT_IDS: tuple[str, ...] = tuple(spec.id for spec in DECANT_CUP_SPECS)
