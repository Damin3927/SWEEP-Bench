from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_TRAY_COLOR = (0.34, 0.38, 0.46, 1.0)

_CHAMFER_HALF_THICKNESS = 0.0015


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def _cross_chamfer(
    *,
    cell_side: float,
    wall_y: float,
    wall_half_thickness: float,
    chamfer: float,
    top_z: float,
    span_half_x: float,
) -> tuple[list[float], sapien.Pose]:
    inner_y = wall_y + cell_side * wall_half_thickness
    theta = math.pi / 2 + cell_side * math.pi / 4
    pose = sapien.Pose(
        p=[0.0, inner_y - cell_side * chamfer / 2.0, top_z + chamfer / 2.0],
        q=[math.cos(theta / 2), math.sin(theta / 2), 0.0, 0.0],
    )
    return [span_half_x, chamfer * math.sqrt(2) / 2, _CHAMFER_HALF_THICKNESS], pose


def _side_chamfer(
    *,
    cell_side: float,
    wall_x: float,
    wall_half_thickness: float,
    chamfer: float,
    top_z: float,
    span_half_y: float,
) -> tuple[list[float], sapien.Pose]:
    inner_x = wall_x + cell_side * wall_half_thickness
    theta = -(math.pi / 2 + cell_side * math.pi / 4)
    pose = sapien.Pose(
        p=[inner_x - cell_side * chamfer / 2.0, 0.0, top_z + chamfer / 2.0],
        q=[math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0],
    )
    return [chamfer * math.sqrt(2) / 2, span_half_y, _CHAMFER_HALF_THICKNESS], pose


def build_kit_tray(
    scene: ManiSkillScene,
    *,
    n_cells: int,
    cell_inner_half: float,
    wall_half_thickness: float,
    wall_half_height: float,
    floor_half_height: float,
    chamfer: float = 0.0,
    color: tuple[float, float, float, float] = _TRAY_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if n_cells < 2:
        raise ValueError(f"n_cells must be >= 2 so a decoy cell exists, got {n_cells}")
    ch, wht, whh, fhh = cell_inner_half, wall_half_thickness, wall_half_height, floor_half_height
    pitch = 2.0 * (ch + wht)
    outer_half_x = ch + 2.0 * wht
    outer_half_y = n_cells * ch + (n_cells + 1) * wht
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
        add_box([wht, outer_half_y, whh], sapien.Pose(p=[side * (ch + wht), 0.0, wall_z]))

    for k in range(n_cells + 1):
        wall_y = (k - n_cells / 2.0) * pitch
        add_box([ch, wht, whh], sapien.Pose(p=[0.0, wall_y, wall_z]))

    if chamfer > 0.0:
        for k in range(n_cells + 1):
            wall_y = (k - n_cells / 2.0) * pitch
            for cell_side in [1.0] if k == 0 else [-1.0] if k == n_cells else [1.0, -1.0]:
                add_box(
                    *_cross_chamfer(
                        cell_side=cell_side,
                        wall_y=wall_y,
                        wall_half_thickness=wht,
                        chamfer=chamfer,
                        top_z=top_z,
                        span_half_x=ch,
                    )
                )
        for side in (1.0, -1.0):
            add_box(
                *_side_chamfer(
                    cell_side=-side,
                    wall_x=side * (ch + wht),
                    wall_half_thickness=wht,
                    chamfer=chamfer,
                    top_z=top_z,
                    span_half_y=outer_half_y,
                )
            )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
