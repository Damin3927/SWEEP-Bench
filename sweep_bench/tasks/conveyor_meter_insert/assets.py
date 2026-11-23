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
_SQRT2 = math.sqrt(2.0)

_ROD_TIP_COLOR = (0.20, 0.22, 0.26, 1.0)
_ROD_BAND_COLOR = (0.95, 0.35, 0.10, 1.0)
_ROD_SHAFT_COLOR = (0.88, 0.90, 0.93, 1.0)
_BLOCK_COLOR = (0.45, 0.55, 0.62, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def _segment(
    builder: ActorBuilder,
    *,
    radius: float,
    z_lo: float,
    z_hi: float,
    color: tuple[float, float, float, float],
) -> None:
    half = (z_hi - z_lo) / 2.0
    if half <= 0.0:
        raise ValueError(f"segment must have positive height (got z_lo={z_lo}, z_hi={z_hi})")
    pose = sapien.Pose(p=[0.0, 0.0, (z_lo + z_hi) / 2.0], q=_UPRIGHT.q)
    builder.add_cylinder_visual(pose=pose, radius=radius, half_length=half, material=_material(color))


def build_banded_rod(
    scene: ManiSkillScene,
    *,
    radius: float,
    half_length: float,
    band_offset: float,
    band_height: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if band_offset <= 0.0:
        raise ValueError(f"band_offset must be positive (got {band_offset})")
    if band_offset + band_height >= 2.0 * half_length:
        raise ValueError(
            f"band (offset {band_offset} + height {band_height}) must fit within the rod length {2 * half_length}"
        )
    builder = scene.create_actor_builder()
    builder.add_cylinder_collision(pose=_UPRIGHT, radius=radius, half_length=half_length)

    tip_z = -half_length
    band_lo = tip_z + band_offset
    band_hi = band_lo + band_height
    _segment(builder, radius=radius, z_lo=tip_z, z_hi=band_lo, color=_ROD_TIP_COLOR)
    _segment(builder, radius=radius, z_lo=band_lo, z_hi=band_hi, color=_ROD_BAND_COLOR)
    _segment(builder, radius=radius, z_lo=band_hi, z_hi=half_length, color=_ROD_SHAFT_COLOR)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_deep_bore_block(
    scene: ManiSkillScene,
    *,
    bore_half_width: float,
    bore_depth: float,
    floor_thickness: float,
    block_half_extent: float,
    chamfer: float = 0.0,
    color: tuple[float, float, float, float] = _BLOCK_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    s, r_out = bore_half_width, block_half_extent
    if not s < r_out:
        raise ValueError(f"bore_half_width={s} must be smaller than block_half_extent={r_out}")
    if bore_depth <= 0.0 or floor_thickness <= 0.0:
        raise ValueError(f"bore_depth={bore_depth} and floor_thickness={floor_thickness} must be positive")

    h = (bore_depth + floor_thickness) / 2.0
    wall_half_h = bore_depth / 2.0
    wall_z = h - wall_half_h
    thickness = (r_out - s) * 0.5
    offset = thickness + s
    mat = _material(color)

    builder = scene.create_actor_builder()
    floor_pose = sapien.Pose(p=[0.0, 0.0, -h + floor_thickness / 2.0])
    builder.add_box_collision(pose=floor_pose, half_size=(r_out, r_out, floor_thickness / 2.0))
    builder.add_box_visual(pose=floor_pose, half_size=(r_out, r_out, floor_thickness / 2.0), material=mat)

    walls = [
        ((thickness, r_out, wall_half_h), [offset, 0.0, wall_z]),
        ((thickness, r_out, wall_half_h), [-offset, 0.0, wall_z]),
        ((r_out, thickness, wall_half_h), [0.0, offset, wall_z]),
        ((r_out, thickness, wall_half_h), [0.0, -offset, wall_z]),
    ]
    for half_size, p in walls:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=half_size)
        builder.add_box_visual(pose=pose, half_size=half_size, material=mat)

    if chamfer > 0.0:
        _add_chamfer(builder, bore_half_width=s, block_half_extent=r_out, top_z=h, chamfer=chamfer, mat=mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def _add_chamfer(
    builder: ActorBuilder,
    *,
    bore_half_width: float,
    block_half_extent: float,
    top_z: float,
    chamfer: float,
    mat: sapien.render.RenderMaterial,
) -> None:
    s, r_out, c = bore_half_width, block_half_extent, chamfer
    half_slope = c / _SQRT2
    z_c = top_z + c / 2
    xy_c = s + c / 2
    plate_thickness = 0.002
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
