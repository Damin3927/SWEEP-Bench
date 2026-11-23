from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_CORE_COLOR = (0.42, 0.40, 0.36, 1.0)
_PANEL_COLOR = (0.85, 0.45, 0.16, 1.0)

CORE_FRICTION = 0.05

PANEL_FRICTION = 2.0


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, metallic=0.1)


def build_shell_core(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_z: float,
    wall_thickness: float,
    web_half_x: float,
    density: float,
    friction: float = CORE_FRICTION,
    color: tuple[float, float, float, float] = _CORE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not web_half_x < half_x:
        raise ValueError(f"web_half_x={web_half_x} must be smaller than half_x={half_x}")
    inner_y = half_y - wall_thickness
    inner_z = half_z - wall_thickness
    if not (inner_y > 0.0 and inner_z > 0.0):
        raise ValueError(f"wall_thickness={wall_thickness} leaves no pocket inside {half_y} x {half_z}")

    mat = _material(color)
    phys = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    builder = scene.create_actor_builder()

    half_wall = 0.5 * wall_thickness
    plates = [
        ((half_x, half_y, half_wall), [0.0, 0.0, -half_z + half_wall]),
        ((half_x, half_y, half_wall), [0.0, 0.0, half_z - half_wall]),
        ((half_x, half_wall, inner_z), [0.0, half_y - half_wall, 0.0]),
        ((half_x, half_wall, inner_z), [0.0, -half_y + half_wall, 0.0]),
        ((web_half_x, inner_y, inner_z), [0.0, 0.0, 0.0]),
    ]
    for half_size, p in plates:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=list(half_size), density=density, material=phys)
        builder.add_box_visual(pose=pose, half_size=list(half_size), material=mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_shell_panel(
    scene: ManiSkillScene,
    *,
    half_thickness: float,
    half_y: float,
    half_z: float,
    density: float,
    friction: float = PANEL_FRICTION,
    color: tuple[float, float, float, float] = _PANEL_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not half_thickness > 0.0:
        raise ValueError(f"half_thickness={half_thickness} must be positive")
    phys = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[half_thickness, half_y, half_z], density=density, material=phys)
    builder.add_box_visual(half_size=[half_thickness, half_y, half_z], material=_material(color))
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
