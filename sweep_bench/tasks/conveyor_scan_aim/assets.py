from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_TO_LOCAL_Z = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])

_HANDLE_COLOR = (0.22, 0.24, 0.28, 1.0)
_MUZZLE_COLOR = (0.98, 0.52, 0.10, 1.0)

_BLOCK_COLOR = (0.58, 0.62, 0.66, 1.0)
_QUIET_COLOR = (0.97, 0.97, 0.95, 1.0)
_CODE_COLOR = (0.06, 0.07, 0.10, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def build_scanner_wand(
    scene: ManiSkillScene,
    *,
    handle_half_width: float,
    handle_half_length: float,
    muzzle_radius: float,
    muzzle_half_length: float,
    handle_color: tuple[float, float, float, float] = _HANDLE_COLOR,
    muzzle_color: tuple[float, float, float, float] = _MUZZLE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if muzzle_radius >= handle_half_width:
        raise ValueError(
            f"muzzle_radius={muzzle_radius} must be smaller than handle_half_width={handle_half_width} "
            "(the snout must be the narrower end so the fingers cannot grip it)"
        )
    builder = scene.create_actor_builder()

    handle_mat = _material(handle_color)
    handle_half = [handle_half_width, handle_half_width, handle_half_length]
    builder.add_box_collision(half_size=handle_half)
    builder.add_box_visual(half_size=handle_half, material=handle_mat)

    muzzle_pose = sapien.Pose(p=[0.0, 0.0, handle_half_length + muzzle_half_length], q=_TO_LOCAL_Z.q)
    muzzle_mat = _material(muzzle_color)
    builder.add_cylinder_collision(pose=muzzle_pose, radius=muzzle_radius, half_length=muzzle_half_length)
    builder.add_cylinder_visual(
        pose=muzzle_pose, radius=muzzle_radius, half_length=muzzle_half_length, material=muzzle_mat
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_code_block(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_z: float,
    patch_half: float,
    patch_local_z: float,
    patch_half_thickness: float = 0.0006,
    block_color: tuple[float, float, float, float] = _BLOCK_COLOR,
    quiet_color: tuple[float, float, float, float] = _QUIET_COLOR,
    code_color: tuple[float, float, float, float] = _CODE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if patch_half + abs(patch_local_z) > half_z:
        raise ValueError(
            f"patch_half={patch_half} at patch_local_z={patch_local_z} runs off the block's "
            f"{half_z} m half-height; the whole patch must lie on the face"
        )
    if patch_half >= half_y:
        raise ValueError(f"patch_half={patch_half} must be smaller than half_y={half_y}")

    builder = scene.create_actor_builder()
    body_half = [half_x, half_y, half_z]
    builder.add_box_collision(half_size=body_half)
    builder.add_box_visual(half_size=body_half, material=_material(block_color))

    quiet_pose = sapien.Pose(p=[-half_x, 0.0, patch_local_z])
    builder.add_box_visual(
        pose=quiet_pose,
        half_size=[patch_half_thickness, patch_half, patch_half],
        material=_material(quiet_color),
    )
    code_pose = sapien.Pose(p=[-half_x - patch_half_thickness, 0.0, patch_local_z])
    builder.add_box_visual(
        pose=code_pose,
        half_size=[patch_half_thickness, 0.6 * patch_half, 0.6 * patch_half],
        material=_material(code_color),
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
