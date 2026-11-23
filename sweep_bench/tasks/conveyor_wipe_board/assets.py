from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_UPRIGHT = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])

_BOARD_FRAME_COLOR = (0.24, 0.26, 0.30, 1.0)
_BOARD_FACE_COLOR = (0.88, 0.88, 0.85, 1.0)
_MARK_COLOR = (0.86, 0.18, 0.16, 1.0)
_PAD_HANDLE_COLOR = (0.82, 0.76, 0.62, 1.0)
_PAD_FACE_COLOR = (0.22, 0.30, 0.44, 1.0)

_MARK_SINK = 0.0005


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def build_marker_board(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_thickness: float,
    mark_radius: float,
    mark_offsets: Sequence[tuple[float, float]],
    density: float,
    frame_color: tuple[float, float, float, float] = _BOARD_FRAME_COLOR,
    face_color: tuple[float, float, float, float] = _BOARD_FACE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not mark_offsets:
        raise ValueError("a marker board needs at least one mark")
    slab_pose = sapien.Pose(p=[0.0, 0.0, -half_thickness])
    builder = scene.create_actor_builder()
    builder.add_box_collision(pose=slab_pose, half_size=[half_x, half_y, half_thickness], density=density)
    builder.add_box_visual(pose=slab_pose, half_size=[half_x, half_y, half_thickness], material=_material(frame_color))

    inset = min(0.006, 0.4 * min(half_x, half_y))
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, -_MARK_SINK]),
        half_size=[half_x - inset, half_y - inset, _MARK_SINK],
        material=_material(face_color),
    )

    for mx, my in mark_offsets:
        if abs(mx) + mark_radius > half_x - inset or abs(my) + mark_radius > half_y - inset:
            raise ValueError(f"mark at ({mx}, {my}) with radius {mark_radius} falls off the writing surface")

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def mark_wear_colors(tiers: int) -> tuple[tuple[float, float, float, float], ...]:
    if tiers < 1:
        raise ValueError(f"tiers={tiers} must be at least 1")
    out = []
    for i in range(tiers):
        t = i / tiers
        out.append(tuple(m * (1.0 - t) + f * t for m, f in zip(_MARK_COLOR, _BOARD_FACE_COLOR, strict=True)))
    return tuple(out)


def build_mark_decal(
    scene: ManiSkillScene,
    *,
    radius: float,
    color: tuple[float, float, float, float],
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    builder.add_cylinder_visual(
        pose=sapien.Pose(p=[0.0, 0.0, 0.0], q=_UPRIGHT.q),
        radius=radius,
        half_length=_MARK_SINK * 0.5,
        material=_material(color),
    )
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_wipe_pad(
    scene: ManiSkillScene,
    *,
    stem_half_grip: float,
    stem_half_long: float,
    stem_above_grip: float,
    stem_below_grip: float,
    face_half_across: float,
    face_half_along: float,
    face_half_thickness: float,
    density: float,
    handle_color: tuple[float, float, float, float] = _PAD_HANDLE_COLOR,
    face_color: tuple[float, float, float, float] = _PAD_FACE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not stem_half_grip < face_half_across:
        raise ValueError(f"stem_half_grip={stem_half_grip} must be smaller than face_half_across={face_half_across}")
    builder = scene.create_actor_builder()

    stem_half_axis = (stem_above_grip + stem_below_grip) * 0.5
    stem_pose = sapien.Pose(p=[(stem_below_grip - stem_above_grip) * 0.5, 0.0, 0.0])
    stem_half = [stem_half_axis, stem_half_long, stem_half_grip]
    builder.add_box_collision(pose=stem_pose, half_size=stem_half, density=density)
    builder.add_box_visual(pose=stem_pose, half_size=stem_half, material=_material(handle_color))

    face_pose = sapien.Pose(p=[stem_below_grip + face_half_thickness, 0.0, 0.0])
    face_half = [face_half_thickness, face_half_along, face_half_across]
    builder.add_box_collision(pose=face_pose, half_size=face_half, density=density)
    builder.add_box_visual(pose=face_pose, half_size=face_half, material=_material(face_color))

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
