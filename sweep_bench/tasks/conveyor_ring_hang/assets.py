from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_BASE_COLOR = (0.45, 0.47, 0.50, 1.0)
_POST_COLOR = (0.90, 0.55, 0.15, 1.0)

_UPRIGHT = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.55)


def build_ring_stand(
    scene: ManiSkillScene,
    *,
    base_half_extent: float,
    base_half_height: float,
    post_radius: float,
    post_height: float,
    base_color: tuple[float, float, float, float] = _BASE_COLOR,
    post_color: tuple[float, float, float, float] = _POST_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if post_radius >= base_half_extent:
        raise ValueError(f"post_radius={post_radius} must be smaller than base_half_extent={base_half_extent}")
    builder = scene.create_actor_builder()

    base_half = [base_half_extent, base_half_extent, base_half_height]
    builder.add_box_collision(half_size=base_half)
    builder.add_box_visual(half_size=base_half, material=_material(base_color))

    post_pose = sapien.Pose(p=[0.0, 0.0, base_half_height + post_height / 2.0], q=_UPRIGHT.q)
    builder.add_cylinder_collision(pose=post_pose, radius=post_radius, half_length=post_height / 2.0)
    builder.add_cylinder_visual(
        pose=post_pose, radius=post_radius, half_length=post_height / 2.0, material=_material(post_color)
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
