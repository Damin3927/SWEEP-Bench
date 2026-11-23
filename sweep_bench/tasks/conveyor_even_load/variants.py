from __future__ import annotations

import functools

from sweep_bench.assets.catalog import ObjectSpec
from sweep_bench.assets.procedural import build_box

BAR_HALF_LENGTH = 0.090
BAR_HALF_WIDTH = 0.020
BAR_HALF_THICKNESS = 0.012
_BAR_COLOR = (0.72, 0.74, 0.78, 1.0)

BAR_DENSITY = 1000.0

EVEN_LOAD_BAR_SPEC = ObjectSpec(
    id="transfer_bar",
    role="graspable_rider",
    source="procedural",
    builder_fn=functools.partial(
        build_box,
        half_x=BAR_HALF_WIDTH,
        half_y=BAR_HALF_LENGTH,
        half_z=BAR_HALF_THICKNESS,
        color=_BAR_COLOR,
    ),
    bbox=(BAR_HALF_WIDTH, BAR_HALF_LENGTH, BAR_HALF_THICKNESS),
    mass=BAR_DENSITY * (2 * BAR_HALF_WIDTH) * (2 * BAR_HALF_LENGTH) * (2 * BAR_HALF_THICKNESS),
    patch_radius=BAR_HALF_WIDTH,
    grasp_width=2 * BAR_HALF_WIDTH,
    tags=("procedural", "box", "bar"),
)
