from __future__ import annotations

import functools

from sweep_bench.assets.catalog import ObjectSpec
from sweep_bench.assets.procedural import build_box

_BLOCK_HALF_XY = 0.0275
_BLOCK_HALF_Z = 0.020
_BLOCK_COLOR = (0.20, 0.45, 0.85, 1.0)

BALANCE_BLOCK_SPEC = ObjectSpec(
    id="block",
    role="graspable_rider",
    source="procedural",
    builder_fn=functools.partial(
        build_box,
        half_x=_BLOCK_HALF_XY,
        half_y=_BLOCK_HALF_XY,
        half_z=_BLOCK_HALF_Z,
        color=_BLOCK_COLOR,
    ),
    bbox=(_BLOCK_HALF_XY, _BLOCK_HALF_XY, _BLOCK_HALF_Z),
    mass=1000.0 * (2 * _BLOCK_HALF_XY) ** 2 * (2 * _BLOCK_HALF_Z),
    patch_radius=_BLOCK_HALF_XY,
    grasp_width=2 * _BLOCK_HALF_XY,
    tags=("procedural", "box", "cube", "blue"),
)

RAIL_BALANCE_OBJECT_IDS: tuple[str, ...] = (
    "block",
    "proc_box_050x050x050_yellow",
    "proc_box_050x050x020_red",
    "proc_box_060x060x060_red",
)

RAIL_BALANCE_REJECTED_RECT_ID: str = "proc_box_070x040x020_red"
RAIL_BALANCE_REJECTED_NARROW_ID: str = "proc_box_040x040x040_green"
