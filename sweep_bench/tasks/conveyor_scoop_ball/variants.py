from __future__ import annotations

from functools import partial

from sweep_bench.assets.catalog import ObjectSpec
from sweep_bench.tasks.conveyor_scoop_ball.assets import build_ball

_BALL_RADII: tuple[tuple[str, float, tuple[float, float, float, float]], ...] = (
    ("proc_ball_048_red", 0.024, (0.85, 0.12, 0.12, 1.0)),
    ("proc_ball_040_yellow", 0.020, (0.90, 0.80, 0.10, 1.0)),
    ("proc_ball_054_blue", 0.027, (0.13, 0.30, 0.75, 1.0)),
)

_BALL_DENSITY = 1000.0


def _ball_spec(object_id: str, radius: float, color: tuple[float, float, float, float]) -> ObjectSpec:
    volume = 4.0 / 3.0 * 3.141592653589793 * radius**3
    return ObjectSpec(
        id=object_id,
        role="graspable_rider",
        source="procedural",
        builder_fn=partial(build_ball, radius=radius, color=color, density=_BALL_DENSITY),
        bbox=(radius, radius, radius),
        mass=_BALL_DENSITY * volume,
        patch_radius=radius,
        grasp_width=None,
        tags=("procedural", "ball", "round", object_id.rsplit("_", 1)[-1]),
    )


BALL_SPECS: tuple[ObjectSpec, ...] = tuple(_ball_spec(*entry) for entry in _BALL_RADII)

SCOOP_BALL_OBJECT_IDS: tuple[str, ...] = tuple(spec.id for spec in BALL_SPECS)
