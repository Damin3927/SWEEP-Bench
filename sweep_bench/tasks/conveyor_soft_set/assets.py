from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_CARRIER_COLOR = (0.38, 0.40, 0.44, 1.0)
_PLATFORM_COLOR = (0.20, 0.55, 0.75, 1.0)

PLATFORM_DENSITY = 100.0

CARRIER_FRICTION = 0.05

PLATFORM_FRICTION = 0.9


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, metallic=0.2)


def build_carrier_well(
    scene: ManiSkillScene,
    *,
    bore_half_width: float,
    outer_half_width: float,
    wall_height: float,
    floor_half_thickness: float,
    friction: float = CARRIER_FRICTION,
    color: tuple[float, float, float, float] = _CARRIER_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    s, r_out, hw, tf = bore_half_width, outer_half_width, wall_height, floor_half_thickness
    if not s < r_out:
        raise ValueError(f"bore_half_width={s} must be smaller than outer_half_width={r_out}")
    mat = _material(color)
    phys = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)

    builder = scene.create_actor_builder()
    floor_pose = sapien.Pose(p=[0.0, 0.0, -tf])
    builder.add_box_collision(pose=floor_pose, half_size=[r_out, r_out, tf], material=phys)
    builder.add_box_visual(pose=floor_pose, half_size=[r_out, r_out, tf], material=mat)

    thickness = (r_out - s) * 0.5
    offset = thickness + s
    walls = [
        ((thickness, r_out, hw * 0.5), [offset, 0.0, hw * 0.5]),
        ((thickness, r_out, hw * 0.5), [-offset, 0.0, hw * 0.5]),
        ((r_out, thickness, hw * 0.5), [0.0, offset, hw * 0.5]),
        ((r_out, thickness, hw * 0.5), [0.0, -offset, hw * 0.5]),
    ]
    for half_size, p in walls:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=half_size, material=phys)
        builder.add_box_visual(pose=pose, half_size=half_size, material=mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_soft_platform(
    scene: ManiSkillScene,
    *,
    plate_half_width: float,
    plate_half_thickness: float,
    density: float = PLATFORM_DENSITY,
    friction: float = PLATFORM_FRICTION,
    color: tuple[float, float, float, float] = _PLATFORM_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    phw, pht = plate_half_width, plate_half_thickness
    if not pht > 0.0:
        raise ValueError(f"plate_half_thickness={pht} must be positive")
    phys = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[phw, phw, pht], density=density, material=phys)
    builder.add_box_visual(half_size=[phw, phw, pht], material=_material(color))
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
