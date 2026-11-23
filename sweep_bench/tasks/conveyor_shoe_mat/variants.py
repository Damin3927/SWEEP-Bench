from __future__ import annotations

import functools
import math

from sweep_bench.assets.catalog import ObjectSpec
from sweep_bench.tasks.conveyor_shoe_mat.assets import build_shoe

_SOLE_COLOR = (0.82, 0.74, 0.58, 1.0)

_BODY_COLORS: dict[str, tuple[float, float, float, float]] = {
    "navy": (0.13, 0.19, 0.42, 1.0),
    "rust": (0.66, 0.27, 0.15, 1.0),
    "olive": (0.36, 0.42, 0.20, 1.0),
}

_HALF_X = 0.025
_HALF_Y = 0.011
_SOLE_HALF_Z = 0.004
_BODY_HALF_Z = 0.016
_COLLAR_HALF_Z = 0.009
_COLLAR_HALF_X = 0.017

_HALF_Z = _SOLE_HALF_Z + _BODY_HALF_Z + _COLLAR_HALF_Z


def _shoe_spec(object_id: str, color_name: str) -> ObjectSpec:
    return ObjectSpec(
        id=object_id,
        role="graspable_rider",
        source="procedural",
        builder_fn=functools.partial(
            build_shoe,
            half_x=_HALF_X,
            half_y=_HALF_Y,
            sole_half_z=_SOLE_HALF_Z,
            body_half_z=_BODY_HALF_Z,
            collar_half_z=_COLLAR_HALF_Z,
            collar_half_x=_COLLAR_HALF_X,
            body_color=_BODY_COLORS[color_name],
            sole_color=_SOLE_COLOR,
        ),
        bbox=(_HALF_X, _HALF_Y, _HALF_Z),
        mass=1000.0 * 8.0 * _HALF_Y * (_HALF_X * (_SOLE_HALF_Z + _BODY_HALF_Z) + _COLLAR_HALF_X * _COLLAR_HALF_Z),
        patch_radius=0.016,
        grasp_width=2.0 * _HALF_X,
        tags=("procedural", "shoe", color_name),
    )


SHOE_SPECS: tuple[ObjectSpec, ...] = (
    _shoe_spec("shoe_navy", "navy"),
    _shoe_spec("shoe_rust", "rust"),
    _shoe_spec("shoe_olive", "olive"),
)

SHOE_MAT_OBJECT_IDS: tuple[str, ...] = tuple(spec.id for spec in SHOE_SPECS)


def sole_tip_angle_deg() -> float:
    boxes = (
        (_HALF_X * _SOLE_HALF_Z, _SOLE_HALF_Z),
        (_HALF_X * _BODY_HALF_Z, 2.0 * _SOLE_HALF_Z + _BODY_HALF_Z),
        (_COLLAR_HALF_X * _COLLAR_HALF_Z, 2.0 * _HALF_Z - _COLLAR_HALF_Z),
    )
    area = sum(a for a, _ in boxes)
    com_z = sum(a * z for a, z in boxes) / area
    return math.degrees(math.atan2(_HALF_Y, com_z))


def shoe_color_name(spec: ObjectSpec) -> str:
    return spec.tags[-1]
