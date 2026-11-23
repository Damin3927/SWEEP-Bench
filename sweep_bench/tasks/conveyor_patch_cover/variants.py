from __future__ import annotations

import functools

from sweep_bench.assets.catalog import ObjectSpec
from sweep_bench.assets.procedural import build_box

_TILE_HALF_Z = 0.010
_FOOTPRINTS: tuple[tuple[float, float], ...] = (
    (0.030, 0.042),
    (0.028, 0.042),
    (0.032, 0.042),
)
_COLORS: dict[str, tuple[float, float, float, float]] = {
    "gray": (0.55, 0.57, 0.60, 1.0),
    "white": (0.90, 0.90, 0.92, 1.0),
}

_DENSITY = 1000.0


def _mm(x: float) -> str:
    return f"{round(x * 1000):03d}"


def _tile_spec(half_x: float, half_y: float, color_name: str) -> ObjectSpec:
    return ObjectSpec(
        id=f"proc_tile_{_mm(2 * half_x)}x{_mm(2 * half_y)}_{color_name}",
        role="graspable_rider",
        source="procedural",
        builder_fn=functools.partial(
            build_box,
            half_x=half_x,
            half_y=half_y,
            half_z=_TILE_HALF_Z,
            color=_COLORS[color_name],
        ),
        bbox=(half_x, half_y, _TILE_HALF_Z),
        mass=_DENSITY * 8 * half_x * half_y * _TILE_HALF_Z,
        patch_radius=min(half_x, half_y),
        grasp_width=2 * half_x,
        tags=("procedural", "tile", "flat", color_name),
    )


PATCH_COVER_TILE_SPECS: tuple[ObjectSpec, ...] = tuple(
    _tile_spec(half_x, half_y, color_name) for half_x, half_y in _FOOTPRINTS for color_name in _COLORS
)

PATCH_COVER_TILE_IDS: tuple[str, ...] = tuple(spec.id for spec in PATCH_COVER_TILE_SPECS)
