from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

Color = tuple[float, float, float, float]

_UPRIGHT = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])


def _material(color: Color) -> sapien.render.RenderMaterial:
    return flat(color)


def _yaw_then_lean(yaw: float, lean: float) -> list[float]:
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    cl, sl = math.cos(0.5 * lean), math.sin(0.5 * lean)
    return [cy * cl, -sy * sl, cy * sl, sy * cl]


def _add_arrow_decal(
    builder: ActorBuilder,
    *,
    tip_x: float,
    tail_x: float,
    half_width: float,
    z: float,
    half_thickness: float,
    color: Color,
    barb_length: float,
    barb_deg: float = 35.0,
) -> None:
    mat = _material(color)
    shaft_half_x = 0.5 * (tip_x - tail_x)
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.5 * (tip_x + tail_x), 0.0, z]),
        half_size=(shaft_half_x, half_width, half_thickness),
        material=mat,
    )
    barb = math.radians(barb_deg)
    for side in (1.0, -1.0):
        angle = math.pi - side * barb
        cx = tip_x + 0.5 * barb_length * math.cos(angle)
        cy = 0.5 * barb_length * math.sin(angle)
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[cx, cy, z],
                q=[math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)],
            ),
            half_size=(0.5 * barb_length, half_width, half_thickness),
            material=mat,
        )


def build_arrow_puck(
    scene: ManiSkillScene,
    *,
    radius: float,
    half_height: float,
    body_color: Color,
    mark_color: Color,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    builder.add_cylinder_collision(pose=_UPRIGHT, radius=radius, half_length=half_height)
    builder.add_cylinder_visual(pose=_UPRIGHT, radius=radius, half_length=half_height, material=_material(body_color))
    _add_arrow_decal(
        builder,
        tip_x=0.92 * radius,
        tail_x=-0.30 * radius,
        half_width=0.14 * radius,
        z=half_height + 0.0008,
        half_thickness=0.0008,
        color=mark_color,
        barb_length=0.42 * radius,
    )
    return builder


def build_round_cradle(
    scene: ManiSkillScene,
    *,
    plate_half_x: float,
    plate_half_y: float,
    plate_half_z: float,
    inner_radius: float,
    wall_half_thickness: float,
    wall_half_height: float,
    wall_tilt_deg: float,
    n_segments: int,
    plate_color: Color,
    wall_color: Color,
) -> ActorBuilder:
    builder = scene.create_actor_builder()

    plate_half = [plate_half_x, plate_half_y, plate_half_z]
    builder.add_box_collision(half_size=plate_half)
    builder.add_box_visual(half_size=plate_half, material=_material(plate_color))

    tilt = math.radians(wall_tilt_deg)
    mat = _material(wall_color)
    hx, hz = wall_half_thickness, wall_half_height
    center_r = inner_radius + hx * math.cos(tilt) + hz * math.sin(tilt)
    center_z = plate_half_z - hx * math.sin(tilt) + hz * math.cos(tilt)
    rim_radius = inner_radius + 2.0 * hz * math.sin(tilt)
    tangential_half = math.pi * rim_radius / n_segments
    for k in range(n_segments):
        theta = 2.0 * math.pi * k / n_segments
        pose = sapien.Pose(
            p=[center_r * math.cos(theta), center_r * math.sin(theta), center_z],
            q=_yaw_then_lean(theta, tilt),
        )
        half = [hx, tangential_half, hz]
        builder.add_box_collision(pose=pose, half_size=half)
        builder.add_box_visual(pose=pose, half_size=half, material=mat)
    return builder


def build_angle_display(
    scene: ManiSkillScene,
    *,
    post_half_width: float,
    post_half_height: float,
    vane_radius: float,
    vane_half_thickness: float,
    post_color: Color,
    vane_color: Color,
    mark_color: Color,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    post_half = [post_half_width, post_half_width, post_half_height]
    builder.add_box_collision(pose=sapien.Pose(p=[0.0, 0.0, -post_half_height]), half_size=post_half)
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, -post_half_height]), half_size=post_half, material=_material(post_color)
    )
    builder.add_cylinder_collision(pose=_UPRIGHT, radius=vane_radius, half_length=vane_half_thickness)
    builder.add_cylinder_visual(
        pose=_UPRIGHT, radius=vane_radius, half_length=vane_half_thickness, material=_material(vane_color)
    )
    _add_arrow_decal(
        builder,
        tip_x=0.92 * vane_radius,
        tail_x=-0.30 * vane_radius,
        half_width=0.14 * vane_radius,
        z=vane_half_thickness + 0.0015,
        half_thickness=0.0015,
        color=mark_color,
        barb_length=0.42 * vane_radius,
    )
    return builder
