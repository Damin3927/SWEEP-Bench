from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_BOX_COLOR = (0.80, 0.72, 0.55, 1.0)
_PATCH_COLOR = (0.90, 0.20, 0.25, 1.0)
_PAD_COLOR = (0.25, 0.27, 0.32, 1.0)
_PAD_FACE_COLOR = (0.20, 0.75, 0.95, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def build_marked_box(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_z: float,
    density: float,
    patch_y: float,
    patch_z: float,
    patch_half_y: float,
    patch_half_z: float,
    patch_half_thickness: float = 0.0003,
    body_color: tuple[float, float, float, float] = _BOX_COLOR,
    patch_color: tuple[float, float, float, float] = _PATCH_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not abs(patch_y) + patch_half_y < half_y:
        raise ValueError(f"patch (y={patch_y}, half={patch_half_y}) does not fit on a face of half_y={half_y}")
    if not abs(patch_z) + patch_half_z < half_z:
        raise ValueError(f"patch (z={patch_z}, half={patch_half_z}) does not fit on a face of half_z={half_z}")

    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[half_x, half_y, half_z], density=density)
    builder.add_box_visual(half_size=[half_x, half_y, half_z], material=_material(body_color))
    builder.add_box_visual(
        pose=sapien.Pose(p=[-(half_x + patch_half_thickness), patch_y, patch_z]),
        half_size=[patch_half_thickness, patch_half_y, patch_half_z],
        material=_material(patch_color),
    )
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_label_pad(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_z: float,
    density: float,
    face_half_thickness: float = 0.0005,
    body_color: tuple[float, float, float, float] = _PAD_COLOR,
    face_color: tuple[float, float, float, float] = _PAD_FACE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[half_x, half_y, half_z], density=density)
    builder.add_box_visual(half_size=[half_x, half_y, half_z], material=_material(body_color))
    builder.add_box_visual(
        pose=sapien.Pose(p=[half_x + face_half_thickness, 0.0, 0.0]),
        half_size=[face_half_thickness, half_y, half_z],
        material=_material(face_color),
    )
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
