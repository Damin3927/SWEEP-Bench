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

_PAD_COLOR = (0.44, 0.46, 0.50, 1.0)
_PLUNGER_BODY_COLOR = (0.28, 0.30, 0.34, 1.0)
_TARGET_COLOR = (0.88, 0.20, 0.14, 1.0)
_STAMP_HANDLE_COLOR = (0.82, 0.76, 0.62, 1.0)
_STAMP_HEAD_COLOR = (0.20, 0.22, 0.28, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def build_seal_pad(
    scene: ManiSkillScene,
    *,
    bore_half_width: float,
    outer_half_width: float,
    collar_height: float,
    floor_half_thickness: float,
    color: tuple[float, float, float, float] = _PAD_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    s, r_out, hc, tf = bore_half_width, outer_half_width, collar_height, floor_half_thickness
    if not s < r_out:
        raise ValueError(f"bore_half_width={s} must be smaller than outer_half_width={r_out}")
    mat = _material(color)

    builder = scene.create_actor_builder()
    floor_pose = sapien.Pose(p=[0.0, 0.0, -tf])
    builder.add_box_collision(pose=floor_pose, half_size=[r_out, r_out, tf])
    builder.add_box_visual(pose=floor_pose, half_size=[r_out, r_out, tf], material=mat)

    thickness = (r_out - s) * 0.5
    offset = thickness + s
    walls = [
        ((thickness, r_out, hc * 0.5), [offset, 0.0, hc * 0.5]),
        ((thickness, r_out, hc * 0.5), [-offset, 0.0, hc * 0.5]),
        ((r_out, thickness, hc * 0.5), [0.0, offset, hc * 0.5]),
        ((r_out, thickness, hc * 0.5), [0.0, -offset, hc * 0.5]),
    ]
    for half_size, p in walls:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=half_size)
        builder.add_box_visual(pose=pose, half_size=half_size, material=mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_seal_plunger(
    scene: ManiSkillScene,
    *,
    half_width: float,
    half_height: float,
    target_radius: float,
    body_color: tuple[float, float, float, float] = _PLUNGER_BODY_COLOR,
    target_color: tuple[float, float, float, float] = _TARGET_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    aw, ah = half_width, half_height
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[aw, aw, ah])
    builder.add_box_visual(half_size=[aw, aw, ah], material=_material(body_color))
    builder.add_cylinder_visual(
        pose=sapien.Pose(p=[0.0, 0.0, ah - 0.001], q=_UPRIGHT.q),
        radius=target_radius,
        half_length=0.001,
        material=_material(target_color),
    )
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_stamp(
    scene: ManiSkillScene,
    *,
    shaft_radius: float,
    shaft_above_grip: float,
    shaft_below_grip: float,
    head_radius: float,
    head_half_thickness: float,
    handle_color: tuple[float, float, float, float] = _STAMP_HANDLE_COLOR,
    head_color: tuple[float, float, float, float] = _STAMP_HEAD_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    r_s, up, down = shaft_radius, shaft_above_grip, shaft_below_grip
    r_h, t_h = head_radius, head_half_thickness
    if not r_s < r_h:
        raise ValueError(f"shaft_radius={r_s} must be smaller than head_radius={r_h}")
    builder = scene.create_actor_builder()

    shaft_half = (up + down) * 0.5
    shaft_pose = sapien.Pose(p=[(down - up) * 0.5, 0.0, 0.0])
    builder.add_cylinder_collision(pose=shaft_pose, radius=r_s, half_length=shaft_half)
    builder.add_cylinder_visual(pose=shaft_pose, radius=r_s, half_length=shaft_half, material=_material(handle_color))

    head_pose = sapien.Pose(p=[down + t_h, 0.0, 0.0])
    builder.add_cylinder_collision(pose=head_pose, radius=r_h, half_length=t_h)
    builder.add_cylinder_visual(pose=head_pose, radius=r_h, half_length=t_h, material=_material(head_color))

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
