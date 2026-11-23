from __future__ import annotations

import functools

from sweep_bench.assets.catalog import ObjectSpec
from sweep_bench.tasks.conveyor_edge_recenter.assets import build_parcel

_PARCEL_HALF_X = 0.028
_PARCEL_HALF_Y = 0.050
_PARCEL_HALF_Z = 0.028
_PARCEL_DENSITY = 400.0
_PARCEL_COLOR = (0.72, 0.55, 0.33, 1.0)

PARCEL_SPEC = ObjectSpec(
    id="parcel",
    role="graspable_rider",
    source="procedural",
    builder_fn=functools.partial(
        build_parcel,
        half_x=_PARCEL_HALF_X,
        half_y=_PARCEL_HALF_Y,
        half_z=_PARCEL_HALF_Z,
        color=_PARCEL_COLOR,
        density=_PARCEL_DENSITY,
    ),
    bbox=(_PARCEL_HALF_X, _PARCEL_HALF_Y, _PARCEL_HALF_Z),
    mass=_PARCEL_DENSITY * 8.0 * _PARCEL_HALF_X * _PARCEL_HALF_Y * _PARCEL_HALF_Z,
    patch_radius=min(_PARCEL_HALF_X, _PARCEL_HALF_Y),
    grasp_width=2 * _PARCEL_HALF_X,
    tags=("procedural", "box", "parcel", "tan"),
)

EDGE_RECENTER_OBJECT_IDS: tuple[str, ...] = (
    "parcel",
    "proc_box_040x040x040_blue",
    "proc_box_050x050x050_green",
    "proc_box_060x060x060_red",
    "proc_box_060x040x030_blue",
)
