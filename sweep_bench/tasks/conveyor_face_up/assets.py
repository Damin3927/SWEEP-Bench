from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

FACE_AXES: tuple[tuple[int, float], ...] = ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0))


def face_normal_world(die_q: torch.Tensor, face_index: torch.Tensor) -> torch.Tensor:
    rot = quaternion_to_matrix(die_q)
    axis = torch.div(face_index, 2, rounding_mode="floor")
    column = torch.gather(rot, 2, axis.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
    sign = torch.where(face_index % 2 == 0, 1.0, -1.0)
    return column * sign.unsqueeze(-1)


def _render_material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def build_marked_die(
    scene: ManiSkillScene,
    *,
    half: float,
    body_color: tuple[float, float, float, float],
    face_colors: tuple[tuple[float, float, float, float], ...],
    face_inset: float = 0.005,
    face_plate_half_thickness: float = 0.0015,
) -> ActorBuilder:
    if len(face_colors) != len(FACE_AXES):
        raise ValueError(f"build_marked_die needs {len(FACE_AXES)} face colours, got {len(face_colors)}")
    if face_inset >= half:
        raise ValueError(f"face_inset={face_inset} does not fit inside a {half} half-extent face")
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[half, half, half])
    builder.add_box_visual(half_size=[half, half, half], material=_render_material(body_color))
    for (axis, sign), color in zip(FACE_AXES, face_colors, strict=True):
        offset = [0.0, 0.0, 0.0]
        offset[axis] = sign * (half + 0.0005)
        plate = [half - face_inset] * 3
        plate[axis] = face_plate_half_thickness
        builder.add_box_visual(
            pose=sapien.Pose(p=offset),
            half_size=plate,
            material=_render_material(color),
        )
    return builder


def build_place_pad(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_z: float,
    paint_half_x: float,
    paint_half_y: float,
    color: tuple[float, float, float, float],
    paint_color: tuple[float, float, float, float],
) -> ActorBuilder:
    if not (0.0 < paint_half_x < half_x and 0.0 < paint_half_y < half_y):
        raise ValueError(
            f"painted footprint ({paint_half_x} x {paint_half_y}) must fit strictly inside the pad "
            f"({half_x} x {half_y}) so a die can rest ON the pad and still be OUTSIDE the footprint"
        )
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[half_x, half_y, half_z])
    builder.add_box_visual(half_size=[half_x, half_y, half_z], material=_render_material(color))
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, half_z + 0.0005]),
        half_size=[paint_half_x, paint_half_y, 0.0005],
        material=_render_material(paint_color),
    )
    return builder
