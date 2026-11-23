from __future__ import annotations

import functools

from sweep_bench.assets.catalog import ObjectSpec
from sweep_bench.tasks.conveyor_key_slot.assets import build_studded_part

PART_HALF_X = 0.0275
PART_HALF_Z = 0.030

PART_STUD_RADIUS = 0.005
PART_STUD_LENGTH = 0.007
PART_HEAD_HALF = 0.011
PART_HEAD_HALF_THICKNESS = 0.003

PART_DENSITY = 800.0
PART_BODY_FRICTION = 0.5
PART_KEY_FRICTION = 0.08

_COLORS: dict[str, tuple[float, float, float, float]] = {
    "blue": (0.18, 0.38, 0.78, 1.0),
    "orange": (0.90, 0.50, 0.10, 1.0),
    "green": (0.15, 0.62, 0.25, 1.0),
    "red": (0.82, 0.15, 0.15, 1.0),
}

_VARIANTS: tuple[tuple[float, str], ...] = (
    (0.0225, "blue"),
    (0.0175, "orange"),
    (0.0250, "green"),
    (0.0225, "red"),
)


def _mm(x: float) -> str:
    return f"{round(x * 1000):03d}"


def _part_spec(half_y: float, color_name: str) -> ObjectSpec:
    body_volume = 8.0 * PART_HALF_X * half_y * PART_HALF_Z
    return ObjectSpec(
        id=f"proc_keyed_part_{_mm(2 * PART_HALF_X)}x{_mm(2 * half_y)}_{color_name}",
        role="graspable_rider",
        source="procedural",
        builder_fn=functools.partial(
            build_studded_part,
            body_half_x=PART_HALF_X,
            body_half_y=half_y,
            body_half_z=PART_HALF_Z,
            stud_radius=PART_STUD_RADIUS,
            stud_length=PART_STUD_LENGTH,
            head_half=PART_HEAD_HALF,
            head_half_thickness=PART_HEAD_HALF_THICKNESS,
            color=_COLORS[color_name],
            density=PART_DENSITY,
            body_friction=PART_BODY_FRICTION,
            key_friction=PART_KEY_FRICTION,
        ),
        bbox=(PART_HALF_X, half_y, PART_HALF_Z),
        mass=PART_DENSITY * body_volume,
        patch_radius=min(PART_HALF_X, half_y),
        grasp_width=2 * PART_HALF_X,
        tags=("procedural", "part", color_name),
    )


KEY_SLOT_PART_SPECS: tuple[ObjectSpec, ...] = tuple(_part_spec(half_y, color) for half_y, color in _VARIANTS)

KEY_SLOT_OBJECT_IDS: tuple[str, ...] = tuple(spec.id for spec in KEY_SLOT_PART_SPECS)

DECOY_HALF_Y: float = _VARIANTS[0][0]
DECOY_COLOR: tuple[float, float, float, float] = (0.45, 0.45, 0.50, 1.0)
