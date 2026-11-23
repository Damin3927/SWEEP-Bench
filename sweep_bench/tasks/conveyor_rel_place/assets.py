from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.physx
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder


def _render_material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def build_long_tray(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    floor_half_height: float,
    pad_half_height: float,
    lip_half_thickness: float,
    lip_half_height: float,
    color: tuple[float, float, float, float],
    pad_color: tuple[float, float, float, float],
    pad_static_friction: float,
    pad_dynamic_friction: float,
) -> ActorBuilder:
    if not 0.0 < pad_half_height < floor_half_height:
        raise ValueError(f"pad_half_height={pad_half_height} must lie strictly inside {floor_half_height}")
    if not 0.0 < 2.0 * lip_half_thickness < min(half_x, half_y):
        raise ValueError(f"lip_half_thickness={lip_half_thickness} leaves no interior in a {half_x} x {half_y} tray")

    builder = scene.create_actor_builder()
    mat, pad_mat = _render_material(color), _render_material(pad_color)

    body_half_z = floor_half_height - pad_half_height
    body_pose = sapien.Pose(p=[0.0, 0.0, -pad_half_height])
    builder.add_box_collision(pose=body_pose, half_size=[half_x, half_y, body_half_z])
    builder.add_box_visual(pose=body_pose, half_size=[half_x, half_y, body_half_z], material=mat)

    pad_pose = sapien.Pose(p=[0.0, 0.0, floor_half_height - pad_half_height])
    pad_material = sapien.physx.PhysxMaterial(
        static_friction=pad_static_friction,
        dynamic_friction=pad_dynamic_friction,
        restitution=0.0,
    )
    builder.add_box_collision(pose=pad_pose, half_size=[half_x, half_y, pad_half_height], material=pad_material)
    builder.add_box_visual(pose=pad_pose, half_size=[half_x, half_y, pad_half_height], material=pad_mat)

    lip_z = floor_half_height + lip_half_height
    inner_half_x = half_x - 2.0 * lip_half_thickness
    for side in (1.0, -1.0):
        pose = sapien.Pose(p=[side * (half_x - lip_half_thickness), 0.0, lip_z])
        half = [lip_half_thickness, half_y, lip_half_height]
        builder.add_box_collision(pose=pose, half_size=half)
        builder.add_box_visual(pose=pose, half_size=half, material=mat)
    for end in (1.0, -1.0):
        pose = sapien.Pose(p=[0.0, end * (half_y - lip_half_thickness), lip_z])
        half = [inner_half_x, lip_half_thickness, lip_half_height]
        builder.add_box_collision(pose=pose, half_size=half)
        builder.add_box_visual(pose=pose, half_size=half, material=mat)

    return builder


def build_anchor_block(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_z: float,
    density: float,
    color: tuple[float, float, float, float],
    static_friction: float,
    dynamic_friction: float,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    material = sapien.physx.PhysxMaterial(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=0.0,
    )
    builder.add_box_collision(half_size=[half_x, half_y, half_z], material=material, density=density)
    builder.add_box_visual(half_size=[half_x, half_y, half_z], material=_render_material(color))
    return builder
