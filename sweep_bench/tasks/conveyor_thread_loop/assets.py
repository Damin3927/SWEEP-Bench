from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_PLATE_COLOR = (0.62, 0.66, 0.70, 1.0)
_POST_COLOR = (0.30, 0.32, 0.36, 1.0)
_RING_COLOR = (0.95, 0.70, 0.10, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.4, metallic=0.3)


def build_loop_carrier(
    scene: ManiSkillScene,
    *,
    plate_half_x: float,
    plate_half_y: float,
    plate_thickness: float,
    post_offset_x: float,
    post_half_thickness: float,
    post_height: float,
    plate_color: tuple[float, float, float, float] = _PLATE_COLOR,
    post_color: tuple[float, float, float, float] = _POST_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if post_offset_x + post_half_thickness > plate_half_x:
        raise ValueError(
            f"posts at |x|={post_offset_x} (half {post_half_thickness}) overhang a plate of half-width {plate_half_x}"
        )
    if post_height <= 0.0 or post_half_thickness <= 0.0 or plate_thickness <= 0.0:
        raise ValueError(
            f"post_height={post_height}, post_half_thickness={post_half_thickness} and "
            f"plate_thickness={plate_thickness} must all be positive"
        )

    builder = scene.create_actor_builder()
    plate_half = [plate_half_x, plate_half_y, plate_thickness / 2.0]
    builder.add_box_collision(half_size=plate_half)
    builder.add_box_visual(half_size=plate_half, material=_material(plate_color))

    post_half = [post_half_thickness, post_half_thickness, post_height / 2.0]
    post_z = plate_thickness / 2.0 + post_height / 2.0
    post_mat = _material(post_color)
    for sign in (1.0, -1.0):
        pose = sapien.Pose(p=[sign * post_offset_x, 0.0, post_z])
        builder.add_box_collision(pose=pose, half_size=post_half)
        builder.add_box_visual(pose=pose, half_size=post_half, material=post_mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_ring(
    scene: ManiSkillScene,
    *,
    inner_radius: float,
    tube_radius: float,
    segments: int = 16,
    density: float = 7800.0,
    color: tuple[float, float, float, float] = _RING_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if segments < 6:
        raise ValueError(f"segments={segments} is too coarse to read as a ring")
    if inner_radius <= 0.0 or tube_radius <= 0.0:
        raise ValueError(f"inner_radius={inner_radius} and tube_radius={tube_radius} must be positive")

    builder = scene.create_actor_builder()
    mat = _material(color)
    mid_radius = inner_radius + tube_radius
    chord_half = mid_radius * math.tan(math.pi / segments)
    tile_half = [chord_half, tube_radius, tube_radius]
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        pose = sapien.Pose(
            p=[mid_radius * math.cos(theta), mid_radius * math.sin(theta), 0.0],
            q=[math.cos((theta - math.pi / 2.0) / 2.0), 0.0, 0.0, math.sin((theta - math.pi / 2.0) / 2.0)],
        )
        builder.add_box_collision(pose=pose, half_size=tile_half, density=density)
        builder.add_box_visual(pose=pose, half_size=tile_half, material=mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
