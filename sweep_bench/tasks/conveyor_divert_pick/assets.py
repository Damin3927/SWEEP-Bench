from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.physx
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_PLOUGH_COLOR = (0.85, 0.55, 0.10, 1.0)
_BLOCK_COLOR = (0.82, 0.80, 0.72, 1.0)
_BLOCK_BAND_COLOR = (0.20, 0.34, 0.52, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def build_divert_plough(
    scene: ManiSkillScene,
    *,
    half_length: float,
    half_thickness: float,
    half_height: float,
    friction: float,
    color: tuple[float, float, float, float] = _PLOUGH_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if half_length <= 0.0 or half_thickness <= 0.0 or half_height <= 0.0:
        raise ValueError(
            f"plough half-extents must all be positive, got ({half_length}, {half_thickness}, {half_height})"
        )
    builder = scene.create_actor_builder()
    material = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    half = [half_thickness, half_length, half_height]
    builder.add_box_collision(half_size=half, material=material)
    builder.add_box_visual(half_size=half, material=_material(color))
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_divert_block(
    scene: ManiSkillScene,
    *,
    half_extent: float,
    density: float,
    friction: float,
    band_frac: float = 0.3,
    body_color: tuple[float, float, float, float] = _BLOCK_COLOR,
    band_color: tuple[float, float, float, float] = _BLOCK_BAND_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not 0.0 < band_frac < 1.0:
        raise ValueError(f"band_frac must be in (0, 1), got {band_frac}")

    builder = scene.create_actor_builder()
    material = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    builder.add_box_collision(half_size=[half_extent] * 3, density=density, material=material)

    band_half = half_extent * band_frac
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, -half_extent + band_half]),
        half_size=[half_extent, half_extent, band_half],
        material=_material(band_color),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, band_half]),
        half_size=[half_extent, half_extent, half_extent - band_half],
        material=_material(body_color),
    )
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
