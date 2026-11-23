from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_HOUSING_COLOR = (0.42, 0.44, 0.48, 1.0)
_ROCKER_COLOR = (0.24, 0.26, 0.30, 1.0)
_PAD_COLOR = (0.87, 0.24, 0.20, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def _add_box(
    builder: ActorBuilder,
    pose: sapien.Pose,
    half_size: tuple[float, float, float],
    mat: sapien.render.RenderMaterial,
) -> None:
    builder.add_box_collision(pose=pose, half_size=half_size)
    builder.add_box_visual(pose=pose, half_size=half_size, material=mat)


def build_rocker_housing(
    scene: ManiSkillScene,
    *,
    well_half_x: float,
    well_half_y: float,
    wall_thickness: float,
    side_rail_height: float,
    end_stop_height: float,
    floor_half_thickness: float,
    fulcrum_height: float,
    color: tuple[float, float, float, float] = _HOUSING_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not fulcrum_height > 0.0:
        raise ValueError(f"fulcrum_height must be positive, got {fulcrum_height}")
    if not end_stop_height > 0.0:
        raise ValueError(f"end_stop_height must be positive, got {end_stop_height}")
    mat = _material(color)
    hx = well_half_x + wall_thickness
    hy = well_half_y + wall_thickness
    tw = wall_thickness * 0.5
    tf = floor_half_thickness

    builder = scene.create_actor_builder()
    _add_box(builder, sapien.Pose(p=[0.0, 0.0, -tf]), (hx, hy, tf), mat)

    for sign in (1.0, -1.0):
        _add_box(
            builder,
            sapien.Pose(p=[0.0, sign * (well_half_y + tw), side_rail_height * 0.5]),
            (hx, tw, side_rail_height * 0.5),
            mat,
        )
        _add_box(
            builder,
            sapien.Pose(p=[sign * (well_half_x + tw), 0.0, end_stop_height * 0.5]),
            (tw, hy, end_stop_height * 0.5),
            mat,
        )

    s = fulcrum_height / math.sqrt(2.0)
    _add_box(
        builder,
        sapien.Pose(p=[0.0, 0.0, 0.0], q=[math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0]),
        (s, well_half_y, s),
        mat,
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_rocker(
    scene: ManiSkillScene,
    *,
    half_length: float,
    half_width: float,
    half_thickness: float,
    pad_offset: float,
    pad_half_x: float,
    pad_half_y: float,
    pad_height: float,
    plate_color: tuple[float, float, float, float] = _ROCKER_COLOR,
    pad_color: tuple[float, float, float, float] = _PAD_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not pad_offset + pad_half_x <= half_length:
        raise ValueError(f"pad (offset {pad_offset} + half {pad_half_x}) overhangs the plate end {half_length}")
    builder = scene.create_actor_builder()
    _add_box(
        builder,
        sapien.Pose(p=[-pad_offset, 0.0, 0.0]),
        (half_length, half_width, half_thickness),
        _material(plate_color),
    )
    pad_mat = _material(pad_color)
    for pad_x in (0.0, -2.0 * pad_offset):
        _add_box(
            builder,
            sapien.Pose(p=[pad_x, 0.0, half_thickness + pad_height * 0.5]),
            (pad_half_x, pad_half_y, pad_height * 0.5),
            pad_mat,
        )
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
