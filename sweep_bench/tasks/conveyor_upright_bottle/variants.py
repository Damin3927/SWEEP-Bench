from __future__ import annotations

import functools

from sweep_bench.assets.catalog import ObjectSpec
from sweep_bench.assets.procedural import build_box

_BOTTLE_DENSITY = 500.0

_BOTTLE_COLORS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.80, 0.16, 0.14, 1.0),
    "blue": (0.15, 0.32, 0.78, 1.0),
    "green": (0.13, 0.58, 0.24, 1.0),
    "yellow": (0.90, 0.78, 0.12, 1.0),
}


def _bottle_spec(shape: str, *, half_long: float, half_short: float, color: str) -> ObjectSpec:
    object_id = f"proc_{shape}_{round(2000 * half_long):03d}x{round(2000 * half_short):03d}_{color}"
    return ObjectSpec(
        id=object_id,
        role="graspable_rider",
        source="procedural",
        builder_fn=functools.partial(
            build_box,
            half_x=half_long,
            half_y=half_short,
            half_z=half_short,
            color=_BOTTLE_COLORS[color],
        ),
        bbox=(half_long, half_short, half_short),
        mass=_BOTTLE_DENSITY * (2 * half_long) * (2 * half_short) ** 2,
        patch_radius=half_short,
        grasp_width=2 * half_short,
        tags=("procedural", shape, color),
    )


BOTTLE_SPECS: tuple[ObjectSpec, ...] = (
    _bottle_spec("bottle", half_long=0.060, half_short=0.020, color="red"),
    _bottle_spec("bottle", half_long=0.060, half_short=0.020, color="blue"),
    _bottle_spec("flask", half_long=0.065, half_short=0.017, color="green"),
    _bottle_spec("jar", half_long=0.055, half_short=0.023, color="yellow"),
)

UPRIGHT_BOTTLE_OBJECT_IDS: tuple[str, ...] = tuple(spec.id for spec in BOTTLE_SPECS)
