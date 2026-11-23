from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_BLOCK_COLOR = (0.62, 0.66, 0.70, 1.0)
_FLARE_COLOR = (0.95, 0.55, 0.20, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def build_side_keyway_block(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_z: float,
    slot_half_z: float,
    slot_half_y: float,
    keyway_depth: float,
    flare_half_z: float,
    flare_depth: float,
    density: float,
    ramp_half_thickness: float = 0.002,
    block_color: tuple[float, float, float, float] = _BLOCK_COLOR,
    flare_color: tuple[float, float, float, float] = _FLARE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not slot_half_z < flare_half_z < half_z:
        raise ValueError(f"need slot_half_z={slot_half_z} < flare_half_z={flare_half_z} < half_z={half_z}")
    if not slot_half_y < half_y:
        raise ValueError(f"slot_half_y={slot_half_y} must be smaller than half_y={half_y}")
    if not flare_depth < keyway_depth < half_x:
        raise ValueError(
            f"keyway_depth={keyway_depth} must sit between flare_depth={flare_depth} and half_x={half_x} "
            "(the two mirrored pockets must leave a solid web between them)"
        )

    body_mat = _material(block_color)
    flare_mat = _material(flare_color)
    builder = scene.create_actor_builder()

    def slab(center: tuple[float, float, float], half: tuple[float, float, float]) -> None:
        pose = sapien.Pose(p=list(center))
        builder.add_box_collision(pose=pose, half_size=list(half), density=density)
        builder.add_box_visual(pose=pose, half_size=list(half), material=body_mat)

    for sz in (-1.0, 1.0):
        slab(
            (0.0, 0.0, sz * (half_z + flare_half_z) / 2.0),
            (half_x, half_y, (half_z - flare_half_z) / 2.0),
        )

    for sz in (-1.0, 1.0):
        slab(
            (0.0, 0.0, sz * (flare_half_z + slot_half_z) / 2.0),
            (half_x - flare_depth, half_y, (flare_half_z - slot_half_z) / 2.0),
        )

    slab((0.0, 0.0, 0.0), (half_x - keyway_depth, half_y, slot_half_z))

    for sy in (-1.0, 1.0):
        slab(
            (0.0, sy * (half_y + slot_half_y) / 2.0, 0.0),
            (half_x, (half_y - slot_half_y) / 2.0, slot_half_z),
        )

    drop = flare_half_z - slot_half_z
    alpha = math.atan2(drop, flare_depth)
    ramp_half = [math.hypot(flare_depth, drop) / 2.0, slot_half_y, ramp_half_thickness]
    mid_x = half_x - flare_depth / 2.0
    mid_z = (flare_half_z + slot_half_z) / 2.0
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            half_angle = -sx * sz * alpha / 2.0
            pose = sapien.Pose(
                p=[
                    sx * (mid_x - ramp_half_thickness * math.sin(alpha)),
                    0.0,
                    sz * (mid_z + ramp_half_thickness * math.cos(alpha)),
                ],
                q=[math.cos(half_angle), 0.0, math.sin(half_angle), 0.0],
            )
            builder.add_box_collision(pose=pose, half_size=ramp_half, density=density)
            builder.add_box_visual(pose=pose, half_size=ramp_half, material=flare_mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
