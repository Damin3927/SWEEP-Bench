from __future__ import annotations

import functools

from sweep_bench.assets.catalog import ObjectSpec
from sweep_bench.tasks.conveyor_read_present.assets import build_marked_part

_BODY_COLOR = (0.55, 0.57, 0.60, 1.0)

_MARK_COLORS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.80, 0.16, 0.16, 1.0),
    "green": (0.16, 0.62, 0.24, 1.0),
    "blue": (0.18, 0.36, 0.82, 1.0),
    "yellow": (0.90, 0.72, 0.10, 1.0),
}

_DEFAULT_HALF = (0.026, 0.020, 0.025)
_WIDE_HALF = (0.030, 0.024, 0.025)


def _marked_part_spec(object_id: str, half: tuple[float, float, float], color_name: str) -> ObjectSpec:
    half_x, half_y, half_z = half
    return ObjectSpec(
        id=object_id,
        role="graspable_rider",
        source="procedural",
        builder_fn=functools.partial(
            build_marked_part,
            half_x=half_x,
            half_y=half_y,
            half_z=half_z,
            body_color=_BODY_COLOR,
            mark_color=_MARK_COLORS[color_name],
        ),
        bbox=half,
        mass=1000.0 * 8.0 * half_x * half_y * half_z,
        patch_radius=min(half_x, half_y),
        grasp_width=2.0 * half_y,
        tags=("procedural", "box", "marked", color_name),
    )


MARKED_PART_SPECS: tuple[ObjectSpec, ...] = (
    _marked_part_spec("marked_part_red", _DEFAULT_HALF, "red"),
    _marked_part_spec("marked_part_green", _DEFAULT_HALF, "green"),
    _marked_part_spec("marked_part_blue", _DEFAULT_HALF, "blue"),
    _marked_part_spec("marked_part_wide_yellow", _WIDE_HALF, "yellow"),
)

READ_PRESENT_OBJECT_IDS: tuple[str, ...] = tuple(spec.id for spec in MARKED_PART_SPECS)


def mark_color_name(spec: ObjectSpec) -> str:
    return spec.tags[-1]
