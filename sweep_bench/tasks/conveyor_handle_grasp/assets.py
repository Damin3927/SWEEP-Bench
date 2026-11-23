from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_ALONG_Z = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])
_ALONG_Y = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])

POST_ROOT_DEPTH = 0.014


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def ring_segment_half_length(body_radius: float, wall_thickness: float) -> float:
    r_mid = body_radius - wall_thickness / 2.0
    return r_mid * math.tan(math.pi / 12.0)


def build_bail_vessel(
    scene: ManiSkillScene,
    *,
    body_radius: float,
    body_half_height: float,
    wall_thickness: float,
    floor_half_thickness: float,
    bar_radius: float,
    aperture: float,
    color: tuple[float, float, float, float],
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if aperture <= 0.0:
        raise ValueError(f"aperture must be positive, got {aperture}")
    if wall_thickness >= body_radius:
        raise ValueError(f"wall_thickness {wall_thickness} must be under body_radius {body_radius}")
    mat = _material(color)
    builder = scene.create_actor_builder()

    def add_box(half: list[float], pose: sapien.Pose) -> None:
        builder.add_box_collision(half_size=half, pose=pose)
        builder.add_box_visual(half_size=half, material=mat, pose=pose)

    def add_cylinder(radius: float, half_length: float, pose: sapien.Pose) -> None:
        builder.add_cylinder_collision(radius=radius, half_length=half_length, pose=pose)
        builder.add_cylinder_visual(radius=radius, half_length=half_length, material=mat, pose=pose)

    r_mid = body_radius - wall_thickness / 2.0
    seg_half = ring_segment_half_length(body_radius, wall_thickness)
    for k in range(12):
        theta = k * math.pi / 6.0
        add_box(
            [wall_thickness / 2.0, seg_half, body_half_height],
            sapien.Pose(
                p=[r_mid * math.cos(theta), r_mid * math.sin(theta), 0.0],
                q=[math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)],
            ),
        )
    floor_pose = sapien.Pose(p=[0.0, 0.0, -body_half_height + floor_half_thickness])
    add_cylinder(body_radius, floor_half_thickness, floor_pose * _ALONG_Z)

    bar_z = body_half_height + aperture + bar_radius
    post_root_z = max(-body_half_height, body_half_height - POST_ROOT_DEPTH)
    post_half = (bar_z - post_root_z) / 2.0
    for sign in (1.0, -1.0):
        post_pose = sapien.Pose(p=[0.0, sign * body_radius, post_root_z + post_half])
        add_cylinder(bar_radius, post_half, post_pose * _ALONG_Z)
    bar_pose = sapien.Pose(p=[0.0, 0.0, bar_z])
    add_cylinder(bar_radius, body_radius, bar_pose * _ALONG_Y)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
