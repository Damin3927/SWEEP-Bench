from __future__ import annotations

from functools import partial

from sweep_bench.assets.catalog import ObjectSpec
from sweep_bench.tasks.conveyor_ramp_roll.assets import build_ball

_PI = 3.141592653589793

_BALL_RADII: tuple[tuple[str, float, tuple[float, float, float, float]], ...] = (
    ("proc_ball_040_amber", 0.020, (0.95, 0.65, 0.15, 1.0)),
    ("proc_ball_044_teal", 0.022, (0.15, 0.60, 0.62, 1.0)),
)

_BALL_DENSITY = 1000.0


def _ball_spec(object_id: str, radius: float, color: tuple[float, float, float, float]) -> ObjectSpec:
    volume = 4.0 / 3.0 * _PI * radius**3
    return ObjectSpec(
        id=object_id,
        role="graspable_rider",
        source="procedural",
        builder_fn=partial(build_ball, radius=radius, color=color, density=_BALL_DENSITY),
        bbox=(radius, radius, radius),
        mass=_BALL_DENSITY * volume,
        patch_radius=radius,
        grasp_width=2.0 * radius,
        tags=("procedural", "ball", "round", object_id.rsplit("_", 1)[-1]),
    )


BALL_SPECS: tuple[ObjectSpec, ...] = tuple(_ball_spec(*entry) for entry in _BALL_RADII)


RAMP_ROLL_OBJECT_IDS: tuple[str, ...] = tuple(spec.id for spec in BALL_SPECS)
