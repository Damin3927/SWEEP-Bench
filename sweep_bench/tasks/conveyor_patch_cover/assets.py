from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.physx
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

PATCH_VISUAL_HALF_Z = 0.0004


def _render_material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def build_patch_board(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_z: float,
    pad_half_height: float,
    pad_static_friction: float,
    pad_dynamic_friction: float,
    color: tuple[float, float, float, float],
    pad_color: tuple[float, float, float, float],
    patch_half_x: float,
    patch_half_y: float,
    patches: Sequence[tuple[float, tuple[float, float, float, float]]],
) -> ActorBuilder:
    if not 0.0 < pad_half_height < half_z:
        raise ValueError(f"pad_half_height={pad_half_height} must lie strictly inside {half_z}")
    for offset, _rgba in patches:
        if abs(offset) + patch_half_y > half_y or patch_half_x > half_x:
            raise ValueError(
                f"patch at y={offset} (half {patch_half_x} x {patch_half_y}) does not fit "
                f"on a {half_x} x {half_y} board"
            )

    builder = scene.create_actor_builder()
    mat, pad_mat = _render_material(color), _render_material(pad_color)

    body_half_z = half_z - pad_half_height
    body_pose = sapien.Pose(p=[0.0, 0.0, -pad_half_height])
    builder.add_box_collision(pose=body_pose, half_size=[half_x, half_y, body_half_z])
    builder.add_box_visual(pose=body_pose, half_size=[half_x, half_y, body_half_z], material=mat)

    pad_pose = sapien.Pose(p=[0.0, 0.0, half_z - pad_half_height])
    pad_material = sapien.physx.PhysxMaterial(
        static_friction=pad_static_friction,
        dynamic_friction=pad_dynamic_friction,
        restitution=0.0,
    )
    builder.add_box_collision(pose=pad_pose, half_size=[half_x, half_y, pad_half_height], material=pad_material)
    builder.add_box_visual(pose=pad_pose, half_size=[half_x, half_y, pad_half_height], material=pad_mat)

    for offset, rgba in patches:
        builder.add_box_visual(
            pose=sapien.Pose(p=[0.0, offset, half_z + PATCH_VISUAL_HALF_Z]),
            half_size=[patch_half_x, patch_half_y, PATCH_VISUAL_HALF_Z],
            material=_render_material(rgba),
        )

    return builder
