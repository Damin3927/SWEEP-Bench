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


def build_tilted_ring_stand(
    scene: ManiSkillScene,
    *,
    base_half_extent: float,
    base_half_height: float,
    post_radius: float,
    post_height: float,
    tilt_rad: float,
    base_color: tuple[float, float, float, float] = _BASE_COLOR,
    post_color: tuple[float, float, float, float] = _POST_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if post_radius >= base_half_extent:
        raise ValueError(f"post_radius={post_radius} must be smaller than base_half_extent={base_half_extent}")
    if not 0.0 <= tilt_rad < math.pi / 4:
        raise ValueError(f"tilt_rad={tilt_rad} must be in [0, pi/4)")
    overhang = post_height * math.sin(tilt_rad)
    if overhang >= base_half_extent:
        raise ValueError(
            f"the tilted post overhangs the base by {overhang:.4f} m, which is outside the "
            f"base_half_extent={base_half_extent}: shorten the post or widen the base"
        )

    builder = scene.create_actor_builder()

    base_half = [base_half_extent, base_half_extent, base_half_height]
    builder.add_box_collision(half_size=base_half)
    builder.add_box_visual(half_size=base_half, material=_material(base_color))

    half = tilt_rad / 2.0
    tilt = sapien.Pose(q=[math.cos(half), 0.0, math.sin(half), 0.0])
    mid = post_height / 2.0
    post_pose = sapien.Pose(
        p=[mid * math.sin(tilt_rad), 0.0, base_half_height + mid * math.cos(tilt_rad)],
        q=(tilt * _UPRIGHT).q,
    )
    builder.add_cylinder_collision(pose=post_pose, radius=post_radius, half_length=mid)
    builder.add_cylinder_visual(pose=post_pose, radius=post_radius, half_length=mid, material=_material(post_color))

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
