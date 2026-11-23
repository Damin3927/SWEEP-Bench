from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_SLEEVE_COLOR = (0.42, 0.44, 0.48, 1.0)
_TRAY_BODY_COLOR = (0.30, 0.32, 0.36, 1.0)
_TRAY_FRONT_COLOR = (0.90, 0.62, 0.16, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def build_drawer_sleeve(
    scene: ManiSkillScene,
    *,
    outer_half_x: float,
    outer_half_y: float,
    channel_half_y: float,
    channel_height: float,
    floor_half_thickness: float,
    top_half_thickness: float,
    back_half_x: float,
    density: float,
    friction: float,
    color: tuple[float, float, float, float] = _SLEEVE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not channel_half_y < outer_half_y:
        raise ValueError(f"channel_half_y={channel_half_y} must be smaller than outer_half_y={outer_half_y}")
    mat = _material(color)
    phys = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)

    builder = scene.create_actor_builder()

    floor_pose = sapien.Pose(p=[0.0, 0.0, -floor_half_thickness])
    builder.add_box_collision(
        pose=floor_pose, half_size=[outer_half_x, outer_half_y, floor_half_thickness], density=density, material=phys
    )
    builder.add_box_visual(pose=floor_pose, half_size=[outer_half_x, outer_half_y, floor_half_thickness], material=mat)

    wall_half_y = (outer_half_y - channel_half_y) * 0.5
    wall_offset = channel_half_y + wall_half_y
    for sy in (-1.0, 1.0):
        pose = sapien.Pose(p=[0.0, sy * wall_offset, channel_height * 0.5])
        builder.add_box_collision(
            pose=pose,
            half_size=[outer_half_x, wall_half_y, channel_height * 0.5],
            density=density,
            material=phys,
        )
        builder.add_box_visual(pose=pose, half_size=[outer_half_x, wall_half_y, channel_height * 0.5], material=mat)

    back_pose = sapien.Pose(p=[outer_half_x - back_half_x, 0.0, channel_height * 0.5])
    builder.add_box_collision(
        pose=back_pose, half_size=[back_half_x, outer_half_y, channel_height * 0.5], density=density, material=phys
    )
    builder.add_box_visual(pose=back_pose, half_size=[back_half_x, outer_half_y, channel_height * 0.5], material=mat)

    top_pose = sapien.Pose(p=[0.0, 0.0, channel_height + top_half_thickness])
    builder.add_box_collision(
        pose=top_pose, half_size=[outer_half_x, outer_half_y, top_half_thickness], density=density, material=phys
    )
    builder.add_box_visual(pose=top_pose, half_size=[outer_half_x, outer_half_y, top_half_thickness], material=mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_drawer_tray(
    scene: ManiSkillScene,
    *,
    body_half_x: float,
    body_half_y: float,
    body_half_z: float,
    front_half_x: float,
    front_half_y: float,
    front_half_z: float,
    front_dz: float,
    density: float,
    body_color: tuple[float, float, float, float] = _TRAY_BODY_COLOR,
    front_color: tuple[float, float, float, float] = _TRAY_FRONT_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[body_half_x, body_half_y, body_half_z], density=density)
    builder.add_box_visual(half_size=[body_half_x, body_half_y, body_half_z], material=_material(body_color))

    front_pose = sapien.Pose(p=[-(body_half_x + front_half_x), 0.0, front_dz])
    builder.add_box_collision(pose=front_pose, half_size=[front_half_x, front_half_y, front_half_z], density=density)
    builder.add_box_visual(
        pose=front_pose, half_size=[front_half_x, front_half_y, front_half_z], material=_material(front_color)
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
