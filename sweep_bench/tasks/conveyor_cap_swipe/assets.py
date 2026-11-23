from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.physx
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_KNOB_COLOR = (0.88, 0.32, 0.16, 1.0)
_FLANGE_COLOR = (0.93, 0.90, 0.84, 1.0)
_BODY_COLOR = (0.34, 0.38, 0.46, 1.0)
_RIM_COLOR = (0.20, 0.23, 0.29, 1.0)

_UPRIGHT = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def build_seat_block(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    floor_half_height: float,
    seat_half: float,
    seat_depth: float,
    density: float,
    body_color: tuple[float, float, float, float] = _BODY_COLOR,
    rim_color: tuple[float, float, float, float] = _RIM_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if seat_depth <= 0.0:
        raise ValueError(f"seat_depth must be positive, got {seat_depth}")
    a = seat_depth / math.sqrt(2.0)
    ridge_x = seat_half + seat_depth
    if ridge_x + seat_depth > min(half_x, half_y) + 1e-9:
        raise ValueError("seat + chamfer ring does not fit inside the carrier footprint")

    body_mat = _material(body_color)
    rim_mat = _material(rim_color)
    builder = scene.create_actor_builder()

    floor = [half_x, half_y, floor_half_height]
    builder.add_box_collision(half_size=floor, density=density)
    builder.add_box_visual(half_size=floor, material=body_mat)

    c = math.cos(math.pi / 8.0)
    s = math.sin(math.pi / 8.0)
    roll_y = [c, 0.0, s, 0.0]
    roll_x = [c, s, 0.0, 0.0]
    ridges = [
        ((a, half_y, a), [ridge_x, 0.0, floor_half_height], roll_y),
        ((a, half_y, a), [-ridge_x, 0.0, floor_half_height], roll_y),
        ((seat_half, a, a), [0.0, ridge_x, floor_half_height], roll_x),
        ((seat_half, a, a), [0.0, -ridge_x, floor_half_height], roll_x),
    ]
    for half, p, q in ridges:
        pose = sapien.Pose(p=p, q=q)
        builder.add_box_collision(pose=pose, half_size=list(half), density=0.0)
        builder.add_box_visual(pose=pose, half_size=list(half), material=rim_mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_knob_cap(
    scene: ManiSkillScene,
    *,
    knob_radius: float,
    knob_half_height: float,
    flange_radius: float,
    flange_half_thickness: float,
    density: float,
    friction: float,
    knob_color: tuple[float, float, float, float] = _KNOB_COLOR,
    flange_color: tuple[float, float, float, float] = _FLANGE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if flange_radius <= knob_radius:
        raise ValueError(f"flange_radius ({flange_radius}) must exceed knob_radius ({knob_radius})")

    total_half = flange_half_thickness + knob_half_height
    flange_z = -total_half + flange_half_thickness
    knob_z = -total_half + 2.0 * flange_half_thickness + knob_half_height

    mat = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    builder = scene.create_actor_builder()

    flange_pose = sapien.Pose(p=[0.0, 0.0, flange_z]) * _UPRIGHT
    builder.add_cylinder_collision(
        pose=flange_pose,
        radius=flange_radius,
        half_length=flange_half_thickness,
        density=density,
        material=mat,
    )
    builder.add_cylinder_visual(
        pose=flange_pose,
        radius=flange_radius,
        half_length=flange_half_thickness,
        material=_material(flange_color),
    )

    knob_pose = sapien.Pose(p=[0.0, 0.0, knob_z]) * _UPRIGHT
    builder.add_cylinder_collision(
        pose=knob_pose,
        radius=knob_radius,
        half_length=knob_half_height,
        density=density,
        material=mat,
    )
    builder.add_cylinder_visual(
        pose=knob_pose,
        radius=knob_radius,
        half_length=knob_half_height,
        material=_material(knob_color),
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
