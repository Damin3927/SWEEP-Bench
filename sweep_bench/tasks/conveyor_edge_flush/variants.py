from __future__ import annotations

import functools

from sweep_bench.assets.catalog import ObjectSpec
from sweep_bench.tasks.conveyor_edge_flush.assets import build_slab

_SLAB_COLOR = (0.88, 0.84, 0.72, 1.0)
_SLAB_HALF_X = 0.0225
_SLAB_HALF_Y = 0.040
_SLAB_HALF_Z = 0.012
_SLAB_DENSITY = 1000.0

SLAB_SPEC = ObjectSpec(
    id="slab",
    role="graspable_rider",
    source="procedural",
    builder_fn=functools.partial(
        build_slab,
        half_x=_SLAB_HALF_X,
        half_y=_SLAB_HALF_Y,
        half_z=_SLAB_HALF_Z,
        density=_SLAB_DENSITY,
        color=_SLAB_COLOR,
    ),
    bbox=(_SLAB_HALF_X, _SLAB_HALF_Y, _SLAB_HALF_Z),
    mass=_SLAB_DENSITY * 8 * _SLAB_HALF_X * _SLAB_HALF_Y * _SLAB_HALF_Z,
    patch_radius=_SLAB_HALF_X,
    grasp_width=2 * _SLAB_HALF_Z,
    tags=("procedural", "slab", "book", "cream"),
)

EDGE_FLUSH_OBJECT_IDS: tuple[str, ...] = (
    "slab",
    "proc_box_070x040x020_red",
    "proc_box_070x040x020_green",
    "proc_box_070x040x020_blue",
    "proc_box_060x040x030_red",
    "proc_box_060x040x030_green",
    "proc_box_060x040x030_blue",
)
