from __future__ import annotations

import math

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

_UPRIGHT_Q = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]


def add_defect_band(
    builder,
    *,
    radius: float,
    half_thickness: float,
    top_z: float,
    color: tuple[float, float, float, float],
) -> None:
    material = flat(color, roughness=0.4)
    pose = sapien.Pose(p=[0.0, 0.0, top_z + half_thickness], q=_UPRIGHT_Q)
    builder.add_cylinder_visual(pose=pose, radius=radius, half_length=half_thickness, material=material)
