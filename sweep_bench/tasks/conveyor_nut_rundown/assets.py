from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_UPRIGHT = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])

_PLATE_COLOR = (0.30, 0.34, 0.40, 1.0)
_BOSS_COLOR = (0.55, 0.57, 0.60, 1.0)
_POST_COLOR = (0.72, 0.74, 0.78, 1.0)
_NUT_COLOR = (0.88, 0.72, 0.16, 1.0)
_WING_COLOR = (0.82, 0.60, 0.10, 1.0)

_GRIPPER_ARC_CLEARANCE = 0.028


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.5)


def thread_material(friction: float) -> sapien.physx.PhysxMaterial:
    return sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)


def _add_box(
    builder: ActorBuilder,
    center: tuple[float, float, float],
    half: tuple[float, float, float],
    color: tuple[float, float, float, float],
    density: float,
    material: sapien.physx.PhysxMaterial | None = None,
    q: tuple[float, float, float, float] | None = None,
) -> None:
    pose = sapien.Pose(p=list(center)) if q is None else sapien.Pose(p=list(center), q=list(q))
    builder.add_box_collision(pose=pose, half_size=half, density=density, material=material)
    builder.add_box_visual(pose=pose, half_size=half, material=_material(color))


def build_nut_fixture(
    scene: ManiSkillScene,
    *,
    plate_half_x: float,
    plate_half_y: float,
    plate_height: float,
    boss_radius: float,
    boss_height: float,
    post_radius: float,
    post_height: float,
    density: float,
) -> ActorBuilder:
    if not post_radius < boss_radius < min(plate_half_x, plate_half_y):
        raise ValueError(
            f"fixture radii must increase: post ({post_radius}) < boss ({boss_radius}) < "
            f"plate half ({min(plate_half_x, plate_half_y)})"
        )

    builder = scene.create_actor_builder()

    _add_box(
        builder,
        (0.0, 0.0, 0.5 * plate_height),
        (plate_half_x, plate_half_y, 0.5 * plate_height),
        _PLATE_COLOR,
        density,
    )

    boss_pose = sapien.Pose(p=[0.0, 0.0, plate_height + 0.5 * boss_height], q=_UPRIGHT.q)
    builder.add_cylinder_collision(pose=boss_pose, radius=boss_radius, half_length=0.5 * boss_height, density=density)
    builder.add_cylinder_visual(
        pose=boss_pose, radius=boss_radius, half_length=0.5 * boss_height, material=_material(_BOSS_COLOR)
    )

    post_pose = sapien.Pose(p=[0.0, 0.0, plate_height + boss_height + 0.5 * post_height], q=_UPRIGHT.q)
    builder.add_cylinder_collision(pose=post_pose, radius=post_radius, half_length=0.5 * post_height, density=density)
    builder.add_cylinder_visual(
        pose=post_pose, radius=post_radius, half_length=0.5 * post_height, material=_material(_POST_COLOR)
    )

    return builder


def build_wing_nut(
    scene: ManiSkillScene,
    *,
    bore_half_width: float,
    body_half_width: float,
    body_half_height: float,
    wing_count: int,
    wing_outer_radius: float,
    wing_half_thickness: float,
    wing_half_height: float,
    density: float,
    friction: float,
) -> ActorBuilder:
    if bore_half_width >= body_half_width:
        raise ValueError(f"bore ({bore_half_width}) must be smaller than the nut body ({body_half_width})")
    if wing_count < 2:
        raise ValueError(f"wing_count ({wing_count}) must be at least 2 for a mass-balanced nut")
    free_half_arc = 0.5 * (2.0 * math.pi * wing_outer_radius / wing_count) - wing_half_thickness
    if free_half_arc < _GRIPPER_ARC_CLEARANCE:
        raise ValueError(
            f"{wing_count} wings at radius {wing_outer_radius} leave only {free_half_arc:.4f} m of "
            f"half-arc between them; the closed gripper needs {_GRIPPER_ARC_CLEARANCE} m to descend "
            f"without landing on a wing"
        )

    builder = scene.create_actor_builder()
    material = thread_material(friction)

    span = 0.5 * (body_half_width - bore_half_width)
    mid = 0.5 * (body_half_width + bore_half_width)
    for sx in (-1.0, 1.0):
        _add_box(
            builder,
            (sx * mid, 0.0, 0.0),
            (span, body_half_width, body_half_height),
            _NUT_COLOR,
            density,
            material,
        )
    for sy in (-1.0, 1.0):
        _add_box(
            builder,
            (0.0, sy * mid, 0.0),
            (bore_half_width, span, body_half_height),
            _NUT_COLOR,
            density,
            material,
        )

    inner = body_half_width
    length = wing_outer_radius - inner
    if length <= 0.0:
        raise ValueError(f"wing_outer_radius ({wing_outer_radius}) must exceed the body ({body_half_width})")
    for k in range(wing_count):
        phi = k * 2.0 * math.pi / wing_count
        cos, sin = math.cos(phi), math.sin(phi)
        mid_r = inner + 0.5 * length
        _add_box(
            builder,
            (-mid_r * sin, mid_r * cos, 0.0),
            (wing_half_thickness, 0.5 * length, wing_half_height),
            _WING_COLOR,
            density,
            material,
            q=(math.cos(0.5 * phi), 0.0, 0.0, math.sin(0.5 * phi)),
        )

    return builder
