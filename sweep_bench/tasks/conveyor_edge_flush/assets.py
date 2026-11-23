from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.physx
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_TRAY_COLOR = (0.35, 0.42, 0.40, 1.0)
_RAIL_COLOR = (0.92, 0.62, 0.10, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def build_rail_tray(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    floor_thickness: float,
    rail_thickness: float,
    rail_height: float,
    density: float,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if min(half_x, half_y) <= 0.0:
        raise ValueError(f"tray half-extents must be positive, got ({half_x}, {half_y})")
    if floor_thickness <= 0.0 or rail_thickness <= 0.0 or rail_height <= 0.0:
        raise ValueError(
            f"floor/rail dimensions must be positive, got ({floor_thickness}, {rail_thickness}, {rail_height})"
        )
    if rail_thickness >= 2.0 * half_x:
        raise ValueError(f"rail_thickness={rail_thickness} does not fit inside a {2 * half_x} m wide tray")
    if density <= 0.0:
        raise ValueError(f"density must be positive, got {density}")

    mat = sapien.physx.PhysxMaterial(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )
    builder = scene.create_actor_builder()

    floor_half_z = floor_thickness / 2.0
    floor_pose = sapien.Pose(p=[0.0, 0.0, -floor_half_z])
    floor_half = [half_x, half_y, floor_half_z]
    builder.add_box_collision(pose=floor_pose, half_size=floor_half, density=density, material=mat)
    builder.add_box_visual(pose=floor_pose, half_size=floor_half, material=_material(_TRAY_COLOR))

    rail_half_x = rail_thickness / 2.0
    rail_half_z = rail_height / 2.0
    rail_pose = sapien.Pose(p=[half_x - rail_half_x, 0.0, rail_half_z])
    rail_half = [rail_half_x, half_y, rail_half_z]
    builder.add_box_collision(pose=rail_pose, half_size=rail_half, density=density, material=mat)
    builder.add_box_visual(pose=rail_pose, half_size=rail_half, material=_material(_RAIL_COLOR))

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_slab(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_z: float,
    density: float,
    color: tuple[float, float, float, float],
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    half = [half_x, half_y, half_z]
    builder.add_box_collision(half_size=half, density=density)
    builder.add_box_visual(half_size=half, material=_material(color))
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
