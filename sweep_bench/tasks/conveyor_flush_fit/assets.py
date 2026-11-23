from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_CARRIER_COLOR = (0.40, 0.42, 0.46, 1.0)

_CHAMFER_HALF_THICKNESS = 0.0015

MIN_WELL_DEPTH = 0.006


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def _cross_chamfer(
    *,
    well_side: float,
    wall_y: float,
    wall_half_thickness: float,
    chamfer: float,
    top_z: float,
    span_half_x: float,
) -> tuple[list[float], sapien.Pose]:
    inner_y = wall_y + well_side * wall_half_thickness
    theta = math.pi / 2 + well_side * math.pi / 4
    pose = sapien.Pose(
        p=[0.0, inner_y - well_side * chamfer / 2.0, top_z + chamfer / 2.0],
        q=[math.cos(theta / 2), math.sin(theta / 2), 0.0, 0.0],
    )
    return [span_half_x, chamfer * math.sqrt(2) / 2, _CHAMFER_HALF_THICKNESS], pose


def _side_chamfer(
    *,
    well_side: float,
    wall_x: float,
    wall_half_thickness: float,
    chamfer: float,
    top_z: float,
    span_half_y: float,
) -> tuple[list[float], sapien.Pose]:
    inner_x = wall_x + well_side * wall_half_thickness
    theta = -(math.pi / 2 + well_side * math.pi / 4)
    pose = sapien.Pose(
        p=[inner_x - well_side * chamfer / 2.0, 0.0, top_z + chamfer / 2.0],
        q=[math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0],
    )
    return [chamfer * math.sqrt(2) / 2, span_half_y, _CHAMFER_HALF_THICKNESS], pose


def build_dual_well_carrier(
    scene: ManiSkillScene,
    *,
    depths: tuple[float, float],
    well_inner_half: float,
    wall_half_thickness: float,
    base_half_height: float,
    chamfer: float = 0.0,
    color: tuple[float, float, float, float] = _CARRIER_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if len(depths) != 2:
        raise ValueError(f"the flush-fit carrier has exactly two wells, got {len(depths)} depths")
    if min(depths) < MIN_WELL_DEPTH:
        raise ValueError(f"well depths must be >= {MIN_WELL_DEPTH} m to cage a part, got {depths}")
    wih, wht, bhh = well_inner_half, wall_half_thickness, base_half_height
    pitch = 2.0 * (wih + wht)
    outer_half_x = wih + 2.0 * wht
    outer_half_y = 2.0 * wih + 3.0 * wht
    wall_height = max(depths)
    top_z = bhh + wall_height
    mat = _material(color)

    builder = scene.create_actor_builder()

    def add_box(half: list[float], pose: sapien.Pose | None = None) -> None:
        kwargs = {} if pose is None else {"pose": pose}
        builder.add_box_collision(half_size=half, **kwargs)
        builder.add_box_visual(half_size=half, material=mat, **kwargs)

    add_box([outer_half_x, outer_half_y, bhh])

    for side in (1.0, -1.0):
        add_box([wht, outer_half_y, wall_height / 2.0], sapien.Pose(p=[side * (wih + wht), 0.0, bhh + wall_height / 2]))

    for k in range(3):
        add_box([wih, wht, wall_height / 2.0], sapien.Pose(p=[0.0, (k - 1.0) * pitch, bhh + wall_height / 2]))

    for k, depth in enumerate(depths):
        riser_half = (wall_height - depth) / 2.0
        if riser_half <= 0.0:
            continue
        add_box([wih, wih, riser_half], sapien.Pose(p=[0.0, (k - 0.5) * pitch, bhh + riser_half]))

    if chamfer > 0.0:
        for k in range(3):
            wall_y = (k - 1.0) * pitch
            for well_side in [1.0] if k == 0 else [-1.0] if k == 2 else [1.0, -1.0]:
                add_box(
                    *_cross_chamfer(
                        well_side=well_side,
                        wall_y=wall_y,
                        wall_half_thickness=wht,
                        chamfer=chamfer,
                        top_z=top_z,
                        span_half_x=wih,
                    )
                )
        for side in (1.0, -1.0):
            add_box(
                *_side_chamfer(
                    well_side=-side,
                    wall_x=side * (wih + wht),
                    wall_half_thickness=wht,
                    chamfer=chamfer,
                    top_z=top_z,
                    span_half_y=outer_half_y,
                )
            )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
