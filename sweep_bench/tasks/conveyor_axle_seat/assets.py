from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_PLATE_COLOR = (0.62, 0.66, 0.70, 1.0)
_FORK_COLOR = (0.95, 0.55, 0.20, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def build_vnotch_carrier(
    scene: ManiSkillScene,
    *,
    plate_half_x: float,
    plate_half_y: float,
    plate_thickness: float,
    fork_pitch: float,
    fork_half_width: float,
    fork_thickness: float,
    throat_rise: float,
    throat_width: float,
    throat_wall_height: float,
    flare_width: float,
    flare_depth: float,
    end_stop_height: float,
    ramp_half_thickness: float = 0.002,
    plate_color: tuple[float, float, float, float] = _PLATE_COLOR,
    fork_color: tuple[float, float, float, float] = _FORK_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not 0.0 < throat_width < flare_width:
        raise ValueError(f"need 0 < throat_width < flare_width (got {throat_width}, {flare_width})")
    if not flare_width / 2.0 < fork_half_width:
        raise ValueError(f"flare half-width {flare_width / 2.0} must fit inside fork_half_width={fork_half_width}")
    if flare_depth <= 0.0 or throat_rise <= 0.0 or end_stop_height <= 0.0 or throat_wall_height <= 0.0:
        raise ValueError(
            f"flare_depth={flare_depth}, throat_rise={throat_rise}, throat_wall_height={throat_wall_height} "
            f"and end_stop_height={end_stop_height} must all be positive"
        )
    if fork_pitch / 2.0 + fork_thickness / 2.0 > plate_half_y:
        raise ValueError(f"forks at pitch {fork_pitch} overhang a plate of half-length {plate_half_y}")
    if fork_half_width > plate_half_x:
        raise ValueError(f"fork_half_width={fork_half_width} overhangs plate_half_x={plate_half_x}")

    plate_mat = _material(plate_color)
    fork_mat = _material(fork_color)
    builder = scene.create_actor_builder()

    plate_top = plate_thickness / 2.0
    fork_half_thickness = fork_thickness / 2.0

    plate_half = [plate_half_x, plate_half_y, plate_thickness / 2.0]
    builder.add_box_collision(half_size=plate_half)
    builder.add_box_visual(half_size=plate_half, material=plate_mat)

    base_half = [fork_half_width, fork_half_thickness, throat_rise / 2.0]
    base_z = plate_top + throat_rise / 2.0

    slot_top = plate_top + throat_rise + throat_wall_height
    slot_half_x = (fork_half_width - throat_width / 2.0) / 2.0
    slot_half = [slot_half_x, fork_half_thickness, throat_wall_height / 2.0]
    slot_x = throat_width / 2.0 + slot_half_x
    slot_z = plate_top + throat_rise + throat_wall_height / 2.0

    dr = (flare_width - throat_width) / 2.0
    slope_half = math.hypot(dr, flare_depth) / 2.0
    alpha = math.atan2(flare_depth, dr)
    ramp_x = (throat_width + flare_width) / 4.0 + ramp_half_thickness * math.sin(alpha)
    ramp_z = slot_top + flare_depth / 2.0 - ramp_half_thickness * math.cos(alpha)
    ramp_half = [slope_half, fork_half_thickness, ramp_half_thickness]
    ramp_q_pos = [math.cos(alpha / 2.0), 0.0, -math.sin(alpha / 2.0), 0.0]
    ramp_q_neg = [math.cos(alpha / 2.0), 0.0, math.sin(alpha / 2.0), 0.0]

    shoulder_top = slot_top + flare_depth + end_stop_height
    shoulder_half_x = (fork_half_width - flare_width / 2.0) / 2.0
    shoulder_half_z = (shoulder_top - slot_top) / 2.0
    shoulder_x = flare_width / 2.0 + shoulder_half_x
    shoulder_z = slot_top + shoulder_half_z
    shoulder_half = [shoulder_half_x, fork_half_thickness, shoulder_half_z]

    for sign in (1.0, -1.0):
        fork_y = sign * fork_pitch / 2.0
        shapes: list[tuple[list[float], sapien.Pose]] = [
            (base_half, sapien.Pose(p=[0.0, fork_y, base_z])),
            (slot_half, sapien.Pose(p=[slot_x, fork_y, slot_z])),
            (slot_half, sapien.Pose(p=[-slot_x, fork_y, slot_z])),
            (ramp_half, sapien.Pose(p=[ramp_x, fork_y, ramp_z], q=ramp_q_pos)),
            (ramp_half, sapien.Pose(p=[-ramp_x, fork_y, ramp_z], q=ramp_q_neg)),
            (shoulder_half, sapien.Pose(p=[shoulder_x, fork_y, shoulder_z])),
            (shoulder_half, sapien.Pose(p=[-shoulder_x, fork_y, shoulder_z])),
        ]
        for half, pose in shapes:
            builder.add_box_collision(pose=pose, half_size=half)
            builder.add_box_visual(pose=pose, half_size=half, material=fork_mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
