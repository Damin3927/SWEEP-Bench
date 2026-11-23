from __future__ import annotations

import functools

from sweep_bench.assets.catalog import ObjectSpec
from sweep_bench.assets.procedural import build_box

_RIDER_HALF = (0.020, 0.020, 0.035)
_RIDER_DENSITY = 1000.0
_RIDER_COLORS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.76, 0.18, 0.16, 1.0),
    "green": (0.20, 0.60, 0.26, 1.0),
    "blue": (0.20, 0.36, 0.72, 1.0),
    "yellow": (0.86, 0.70, 0.16, 1.0),
}


def _spec(color_name: str) -> ObjectSpec:
    hx, hy, hz = _RIDER_HALF
    return ObjectSpec(
        id=f"proc_box_040x040x070_{color_name}",
        role="graspable_rider",
        source="procedural",
        builder_fn=functools.partial(build_box, half_x=hx, half_y=hy, half_z=hz, color=_RIDER_COLORS[color_name]),
        bbox=(hx, hy, hz),
        mass=_RIDER_DENSITY * 8.0 * hx * hy * hz,
        patch_radius=min(hx, hy),
        grasp_width=2.0 * min(hx, hy),
        tags=("procedural", "box", "tall", color_name),
    )


GAP_THREAD_RIDER_SPECS: tuple[ObjectSpec, ...] = tuple(_spec(name) for name in _RIDER_COLORS)

GAP_THREAD_OBJECT_IDS: tuple[str, ...] = tuple(spec.id for spec in GAP_THREAD_RIDER_SPECS)
