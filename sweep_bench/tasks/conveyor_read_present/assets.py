from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_POST_COLOR = (0.24, 0.26, 0.30, 1.0)
_FRAME_COLOR = (0.95, 0.55, 0.10, 1.0)


def _render_material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def build_marked_part(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_z: float,
    body_color: tuple[float, float, float, float],
    mark_color: tuple[float, float, float, float],
    mark_inset: float = 0.004,
    mark_half_thickness: float = 0.0015,
) -> ActorBuilder:
    if mark_inset >= min(half_y, half_z):
        raise ValueError(f"mark_inset={mark_inset} does not fit inside the {half_y} x {half_z} face")
    builder = scene.create_actor_builder()
    half = [half_x, half_y, half_z]
    builder.add_box_collision(half_size=half)
    builder.add_box_visual(half_size=half, material=_render_material(body_color))
    builder.add_box_visual(
        pose=sapien.Pose(p=[half_x + 0.0005, 0.0, 0.0]),
        half_size=[mark_half_thickness, half_y - mark_inset, half_z - mark_inset],
        material=_render_material(mark_color),
    )
    return builder


def build_reader_window(
    scene: ManiSkillScene,
    *,
    post_height: float,
    window_half: float,
    frame_bar_half: float,
    frame_half_thickness: float,
    post_half_width: float,
    post_color: tuple[float, float, float, float] = _POST_COLOR,
    frame_color: tuple[float, float, float, float] = _FRAME_COLOR,
) -> ActorBuilder:
    outer_half = window_half + 2.0 * frame_bar_half
    if post_height <= outer_half:
        raise ValueError(f"post_height={post_height} must exceed the frame's outer half-extent {outer_half}")

    builder = scene.create_actor_builder()
    frame_mat = _render_material(frame_color)
    bar_offset = window_half + frame_bar_half

    for sign in (1.0, -1.0):
        pose = sapien.Pose(p=[0.0, 0.0, sign * bar_offset])
        half = [frame_half_thickness, outer_half, frame_bar_half]
        builder.add_box_collision(pose=pose, half_size=half)
        builder.add_box_visual(pose=pose, half_size=half, material=frame_mat)
    for sign in (1.0, -1.0):
        pose = sapien.Pose(p=[0.0, sign * bar_offset, 0.0])
        half = [frame_half_thickness, frame_bar_half, window_half]
        builder.add_box_collision(pose=pose, half_size=half)
        builder.add_box_visual(pose=pose, half_size=half, material=frame_mat)

    post_half_z = 0.5 * (post_height - outer_half)
    post_pose = sapien.Pose(p=[0.0, 0.0, -0.5 * (post_height + outer_half)])
    post_half = [post_half_width, post_half_width, post_half_z]
    builder.add_box_collision(pose=post_pose, half_size=post_half)
    builder.add_box_visual(pose=post_pose, half_size=post_half, material=_render_material(post_color))
    return builder
