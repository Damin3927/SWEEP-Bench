from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import Finish, flat, textured

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_PEG_TIP_COLOR = (0.93, 0.45, 0.34, 1.0)
_PEG_SHAFT_COLOR = (0.93, 0.96, 0.97, 1.0)
_BLOCK_COLOR = (1.0, 0.82, 0.54, 1.0)

_SQRT2 = math.sqrt(2.0)

_CHAMFER_PLATE_THICKNESS = 0.002


def _material(
    color: tuple[float, float, float, float], finish: Finish | None = None, finish_key: str = ""
) -> sapien.render.RenderMaterial:
    return flat(color) if finish is None else textured(color, finish, key=finish_key)


def build_peg(
    scene: ManiSkillScene,
    *,
    radius: float,
    half_length: float,
    tip_color: tuple[float, float, float, float] = _PEG_TIP_COLOR,
    shaft_color: tuple[float, float, float, float] = _PEG_SHAFT_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    builder.add_cylinder_collision(radius=radius, half_length=half_length)
    builder.add_cylinder_visual(
        pose=sapien.Pose(p=[half_length / 2, 0.0, 0.0]),
        radius=radius,
        half_length=half_length / 2,
        material=_material(tip_color),
    )
    builder.add_cylinder_visual(
        pose=sapien.Pose(p=[-half_length / 2, 0.0, 0.0]),
        radius=radius,
        half_length=half_length / 2,
        material=_material(shaft_color),
    )
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_hole_block(
    scene: ManiSkillScene,
    *,
    hole_half_width: float,
    block_half_extent: float,
    block_half_height: float,
    chamfer: float = 0.0,
    color: tuple[float, float, float, float] = _BLOCK_COLOR,
    finish: Finish | None = None,
    finish_key: str = "",
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    s, r_out, h = hole_half_width, block_half_extent, block_half_height
    if not s < r_out:
        raise ValueError(f"hole_half_width={s} must be smaller than block_half_extent={r_out}")
    thickness = (r_out - s) * 0.5
    offset = thickness + s
    mat = _material(color, finish, finish_key)

    builder = scene.create_actor_builder()
    walls = [
        ((thickness, r_out, h), [offset, 0.0, 0.0]),
        ((thickness, r_out, h), [-offset, 0.0, 0.0]),
        ((r_out, thickness, h), [0.0, offset, 0.0]),
        ((r_out, thickness, h), [0.0, -offset, 0.0]),
    ]
    for half_size, p in walls:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=half_size)
        builder.add_box_visual(pose=pose, half_size=half_size, material=mat)

    if chamfer > 0.0:
        _add_chamfer(builder, hole_half_width=s, block_half_extent=r_out, block_half_height=h, chamfer=chamfer, mat=mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def _add_chamfer(
    builder: ActorBuilder,
    *,
    hole_half_width: float,
    block_half_extent: float,
    block_half_height: float,
    chamfer: float,
    mat: sapien.render.RenderMaterial,
) -> None:
    s, r_out, h, c = hole_half_width, block_half_extent, block_half_height, chamfer
    half_slope = c / _SQRT2
    z_c = h + c / 2
    xy_c = s + c / 2
    plate_thickness = _CHAMFER_PLATE_THICKNESS

    cos_h, sin_h = math.cos(math.pi / 8), math.sin(math.pi / 8)
    ramps = [
        ((half_slope, r_out, plate_thickness), [xy_c, 0.0, z_c], [cos_h, 0.0, -sin_h, 0.0]),
        ((half_slope, r_out, plate_thickness), [-xy_c, 0.0, z_c], [cos_h, 0.0, sin_h, 0.0]),
        ((r_out, half_slope, plate_thickness), [0.0, xy_c, z_c], [cos_h, sin_h, 0.0, 0.0]),
        ((r_out, half_slope, plate_thickness), [0.0, -xy_c, z_c], [cos_h, -sin_h, 0.0, 0.0]),
    ]
    for half_size, p, q in ramps:
        pose = sapien.Pose(p=p, q=q)
        builder.add_box_collision(pose=pose, half_size=half_size)
        builder.add_box_visual(pose=pose, half_size=half_size, material=mat)


def hole_block_collision_volume(
    *,
    hole_half_width: float,
    block_half_extent: float,
    block_half_height: float,
    chamfer: float = 0.0,
) -> float:
    s, r_out, h = hole_half_width, block_half_extent, block_half_height
    thickness = (r_out - s) * 0.5
    walls = 4 * (2 * thickness) * (2 * r_out) * (2 * h)
    plates = 4 * (2 * chamfer / _SQRT2) * (2 * r_out) * (2 * _CHAMFER_PLATE_THICKNESS)
    return walls + plates
