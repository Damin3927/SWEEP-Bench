from __future__ import annotations

from typing import TYPE_CHECKING

import sapien

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def build_shoe(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    sole_half_z: float,
    body_half_z: float,
    collar_half_z: float,
    collar_half_x: float,
    body_color: tuple[float, float, float, float],
    sole_color: tuple[float, float, float, float],
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    half_z = sole_half_z + body_half_z + collar_half_z
    for hs, cz, color in (
        ((half_x, half_y, sole_half_z), -half_z + sole_half_z, sole_color),
        ((half_x, half_y, body_half_z), -half_z + 2.0 * sole_half_z + body_half_z, body_color),
        ((collar_half_x, half_y, collar_half_z), half_z - collar_half_z, body_color),
    ):
        pose = sapien.Pose(p=[0.0, 0.0, cz])
        builder.add_box_collision(pose=pose, half_size=list(hs))
        builder.add_box_visual(pose=pose, half_size=list(hs), material=_material(color))
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_mat(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_z: float,
    inlay_margin: float,
    color: tuple[float, float, float, float],
    inlay_color: tuple[float, float, float, float],
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[half_x, half_y, half_z])
    builder.add_box_visual(half_size=[half_x, half_y, half_z], material=_material(color))
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, half_z]),
        half_size=[half_x - inlay_margin, half_y - inlay_margin, 0.0001],
        material=_material(inlay_color),
    )
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
