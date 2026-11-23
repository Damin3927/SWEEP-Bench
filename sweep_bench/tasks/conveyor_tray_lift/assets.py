from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def build_tray(
    scene: ManiSkillScene,
    *,
    inner_half_x: float,
    inner_half_y: float,
    wall_thickness: float,
    wall_half_height: float,
    floor_half_height: float,
    color: tuple[float, float, float, float],
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    wt = wall_thickness
    outer_half_x = inner_half_x + 2 * wt
    outer_half_y = inner_half_y + 2 * wt
    half_z = floor_half_height + wall_half_height
    floor_z = -half_z + floor_half_height
    wall_z = -half_z + 2 * floor_half_height + wall_half_height
    mat = _material(color)

    builder = scene.create_actor_builder()

    def add_box(half: list[float], p: list[float]) -> None:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=half)
        builder.add_box_visual(pose=pose, half_size=half, material=mat)

    add_box([outer_half_x, outer_half_y, floor_half_height], [0.0, 0.0, floor_z])
    for sx in (1.0, -1.0):
        add_box([wt, outer_half_y, wall_half_height], [sx * (inner_half_x + wt), 0.0, wall_z])
    for sy in (1.0, -1.0):
        add_box([inner_half_x, wt, wall_half_height], [0.0, sy * (inner_half_y + wt), wall_z])

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_ball(
    scene: ManiSkillScene,
    *,
    radius: float,
    color: tuple[float, float, float, float],
    density: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    builder.add_sphere_collision(radius=radius, density=density)
    builder.add_sphere_visual(radius=radius, material=_material(color))
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
