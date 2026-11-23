from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_BASE_COLOR = (0.45, 0.47, 0.50, 1.0)
_BAR_COLOR = (0.90, 0.55, 0.15, 1.0)
_SADDLE_COLOR = (0.30, 0.45, 0.72, 1.0)

_ALONG_Y = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])

_PLATE_HALF_THICKNESS = 0.0015
_LAND_HALF_HEIGHT = 0.0015


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.55)


def _tilted_plate(
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    outward: tuple[float, float],
    half_y: float,
    half_thickness: float,
) -> tuple[list[float], sapien.Pose]:
    (x0, z0), (x1, z1) = start, end
    length = math.hypot(x1 - x0, z1 - z0)
    nx, nz = outward
    norm = math.hypot(nx, nz)
    if length <= 0.0 or norm <= 0.0:
        raise ValueError(f"degenerate plate: start={start} end={end} outward={outward}")
    nx, nz = nx / norm, nz / norm
    cx = 0.5 * (x0 + x1) + nx * half_thickness
    cz = 0.5 * (z0 + z1) + nz * half_thickness
    theta = math.atan2(nx, nz)
    q = [math.cos(0.5 * theta), 0.0, math.sin(0.5 * theta), 0.0]
    return [0.5 * length, half_y, half_thickness], sapien.Pose(p=[cx, 0.0, cz], q=q)


def build_hook_rack(
    scene: ManiSkillScene,
    *,
    base_half_x: float,
    base_half_y: float,
    base_half_height: float,
    upright_half_x: float,
    upright_half_y: float,
    upright_half_height: float,
    upright_span_half: float,
    bar_radius: float,
    bar_half_length: float,
    land_half_x: float,
    base_color: tuple[float, float, float, float] = _BASE_COLOR,
    bar_color: tuple[float, float, float, float] = _BAR_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if land_half_x >= bar_radius:
        raise ValueError(f"land_half_x={land_half_x} must be smaller than bar_radius={bar_radius}")
    if upright_span_half + upright_half_y > base_half_y:
        raise ValueError("uprights must stand fully on the base plate in y")
    builder = scene.create_actor_builder()
    base_mat = _material(base_color)
    bar_mat = _material(bar_color)

    base_half = [base_half_x, base_half_y, base_half_height]
    builder.add_box_collision(half_size=base_half)
    builder.add_box_visual(half_size=base_half, material=base_mat)

    upright_half = [upright_half_x, upright_half_y, upright_half_height]
    upright_z = base_half_height + upright_half_height
    for sign in (-1.0, 1.0):
        pose = sapien.Pose(p=[0.0, sign * upright_span_half, upright_z])
        builder.add_box_collision(pose=pose, half_size=upright_half)
        builder.add_box_visual(pose=pose, half_size=upright_half, material=base_mat)

    bar_z = base_half_height + 2.0 * upright_half_height
    bar_pose = sapien.Pose(p=[0.0, 0.0, bar_z], q=_ALONG_Y.q)
    builder.add_cylinder_collision(pose=bar_pose, radius=bar_radius, half_length=bar_half_length)
    builder.add_cylinder_visual(pose=bar_pose, radius=bar_radius, half_length=bar_half_length, material=bar_mat)

    land_pose = sapien.Pose(p=[0.0, 0.0, bar_z + bar_radius - _LAND_HALF_HEIGHT])
    land_half = [land_half_x, bar_half_length, _LAND_HALF_HEIGHT]
    builder.add_box_collision(pose=land_pose, half_size=land_half)
    builder.add_box_visual(pose=land_pose, half_size=land_half, material=bar_mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_rail_saddle(
    scene: ManiSkillScene,
    *,
    seat_local_z: float,
    throat_half_width: float,
    throat_depth: float,
    wall_thickness: float,
    half_y: float,
    cradle_rise: float,
    lip_flare: float,
    lip_height: float,
    roof_half_height: float,
    color: tuple[float, float, float, float] = _SADDLE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if cradle_rise <= 0.0 or cradle_rise >= throat_half_width:
        raise ValueError(f"cradle_rise={cradle_rise} must be in (0, throat_half_width={throat_half_width})")
    builder = scene.create_actor_builder()
    mat = _material(color)

    outer_half_x = throat_half_width + wall_thickness
    apex_z = seat_local_z + cradle_rise
    mouth_z = seat_local_z - throat_depth
    wall_half_x = 0.5 * wall_thickness
    wall_centre_x = throat_half_width + wall_half_x
    wall_half_height = 0.5 * (apex_z - mouth_z)
    wall_centre_z = 0.5 * (apex_z + mouth_z)

    for sign in (-1.0, 1.0):
        pose = sapien.Pose(p=[sign * wall_centre_x, 0.0, wall_centre_z])
        half = [wall_half_x, half_y, wall_half_height]
        builder.add_box_collision(pose=pose, half_size=half)
        builder.add_box_visual(pose=pose, half_size=half, material=mat)

    for sign in (-1.0, 1.0):
        half, pose = _tilted_plate(
            start=(0.0, apex_z),
            end=(sign * cradle_rise, apex_z - cradle_rise),
            outward=(sign, 1.0),
            half_y=half_y,
            half_thickness=_PLATE_HALF_THICKNESS,
        )
        builder.add_box_collision(pose=pose, half_size=half)
        builder.add_box_visual(pose=pose, half_size=half, material=mat)

    for sign in (-1.0, 1.0):
        half, pose = _tilted_plate(
            start=(sign * throat_half_width, mouth_z),
            end=(sign * (throat_half_width + lip_flare), mouth_z - lip_height),
            outward=(sign, lip_flare / lip_height),
            half_y=half_y,
            half_thickness=_PLATE_HALF_THICKNESS,
        )
        builder.add_box_collision(pose=pose, half_size=half)
        builder.add_box_visual(pose=pose, half_size=half, material=mat)

    roof_pose = sapien.Pose(p=[0.0, 0.0, apex_z + roof_half_height])
    roof_half = [outer_half_x, half_y, roof_half_height]
    builder.add_box_collision(pose=roof_pose, half_size=roof_half)
    builder.add_box_visual(pose=roof_pose, half_size=roof_half, material=mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
