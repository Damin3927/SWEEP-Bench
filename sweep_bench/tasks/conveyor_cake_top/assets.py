from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_CAKE_COLOR = (0.93, 0.86, 0.70, 1.0)
_ZONE_COLOR = (0.86, 0.22, 0.40, 1.0)

_DECAL_HALF_THICKNESS = 0.0003

_UPRIGHT = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.7)


def crown_disc_geometry(
    *,
    radius: float,
    body_half_z: float,
    crown_rise: float,
    crown_discs: int,
    plateau_radius: float,
) -> tuple[tuple[float, float], ...]:
    if crown_rise <= 0.0 or crown_discs <= 0:
        return ()
    if not (0.0 < plateau_radius < radius):
        raise ValueError(f"plateau_radius must lie in (0, {radius}), got {plateau_radius}")
    step_r = (radius - plateau_radius) / crown_discs
    step_z = crown_rise / crown_discs
    return tuple((radius - (k + 1) * step_r, body_half_z + (k + 1) * step_z) for k in range(crown_discs))


def build_cake_blank(
    scene: ManiSkillScene,
    *,
    radius: float,
    body_half_z: float,
    crown_rise: float,
    crown_discs: int,
    plateau_radius: float,
    zone_offset: tuple[float, float],
    zone_radius: float,
    color: tuple[float, float, float, float] = _CAKE_COLOR,
    zone_color: tuple[float, float, float, float] = _ZONE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    mat = _material(color)
    builder = scene.create_actor_builder()

    builder.add_cylinder_collision(pose=_UPRIGHT, radius=radius, half_length=body_half_z)
    builder.add_cylinder_visual(pose=_UPRIGHT, radius=radius, half_length=body_half_z, material=mat)

    prev_top = body_half_z
    for disc_radius, disc_top in crown_disc_geometry(
        radius=radius,
        body_half_z=body_half_z,
        crown_rise=crown_rise,
        crown_discs=crown_discs,
        plateau_radius=plateau_radius,
    ):
        half = (disc_top - prev_top) / 2.0
        pose = sapien.Pose(p=[0.0, 0.0, prev_top + half], q=_UPRIGHT.q)
        builder.add_cylinder_collision(pose=pose, radius=disc_radius, half_length=half)
        builder.add_cylinder_visual(pose=pose, radius=disc_radius, half_length=half, material=mat)
        prev_top = disc_top

    decal_pose = sapien.Pose(
        p=[zone_offset[0], zone_offset[1], body_half_z + crown_rise + _DECAL_HALF_THICKNESS],
        q=_UPRIGHT.q,
    )
    builder.add_cylinder_visual(
        pose=decal_pose,
        radius=zone_radius,
        half_length=_DECAL_HALF_THICKNESS,
        material=_material(zone_color),
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
