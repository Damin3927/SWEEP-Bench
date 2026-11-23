from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_TOTE_COLOR = (0.32, 0.40, 0.48, 1.0)

_LIP_HALF_THICKNESS = 0.0015


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def _x_lip(
    *,
    side: float,
    inner_half_x: float,
    lip: float,
    top_z: float,
    span_half_y: float,
) -> tuple[list[float], sapien.Pose]:
    theta = -(math.pi / 2 - side * math.pi / 4)
    pose = sapien.Pose(
        p=[side * (inner_half_x + lip / 2.0), 0.0, top_z + lip / 2.0],
        q=[math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0],
    )
    return [lip * math.sqrt(2) / 2, span_half_y, _LIP_HALF_THICKNESS], pose


def _y_lip(
    *,
    side: float,
    inner_half_y: float,
    lip: float,
    top_z: float,
    span_half_x: float,
) -> tuple[list[float], sapien.Pose]:
    theta = math.pi / 2 - side * math.pi / 4
    pose = sapien.Pose(
        p=[0.0, side * (inner_half_y + lip / 2.0), top_z + lip / 2.0],
        q=[math.cos(theta / 2), math.sin(theta / 2), 0.0, 0.0],
    )
    return [span_half_x, lip * math.sqrt(2) / 2, _LIP_HALF_THICKNESS], pose


def build_tote(
    scene: ManiSkillScene,
    *,
    inner_half_x: float,
    inner_half_y: float,
    wall_half_thickness: float,
    wall_half_height: float,
    floor_half_height: float,
    lip: float = 0.0,
    color: tuple[float, float, float, float] = _TOTE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    ihx, ihy, wht = inner_half_x, inner_half_y, wall_half_thickness
    whh, fhh = wall_half_height, floor_half_height
    outer_half_x = ihx + 2.0 * wht
    outer_half_y = ihy + 2.0 * wht
    wall_z = fhh + whh
    top_z = fhh + 2.0 * whh
    mat = _material(color)

    builder = scene.create_actor_builder()

    def add_box(half: list[float], pose: sapien.Pose | None = None) -> None:
        kwargs = {} if pose is None else {"pose": pose}
        builder.add_box_collision(half_size=half, **kwargs)
        builder.add_box_visual(half_size=half, material=mat, **kwargs)

    add_box([outer_half_x, outer_half_y, fhh])

    for side in (1.0, -1.0):
        add_box([wht, outer_half_y, whh], sapien.Pose(p=[side * (ihx + wht), 0.0, wall_z]))
    for side in (1.0, -1.0):
        add_box([ihx, wht, whh], sapien.Pose(p=[0.0, side * (ihy + wht), wall_z]))

    if lip > 0.0:
        for side in (1.0, -1.0):
            add_box(*_x_lip(side=side, inner_half_x=ihx, lip=lip, top_z=top_z, span_half_y=outer_half_y))
            add_box(*_y_lip(side=side, inner_half_y=ihy, lip=lip, top_z=top_z, span_half_x=ihx))

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
