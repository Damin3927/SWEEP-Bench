from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_HOUSING_COLOR = (0.42, 0.44, 0.48, 1.0)
_PLUNGER_BODY_COLOR = (0.28, 0.30, 0.34, 1.0)
_PLUNGER_FACE_COLOR = (0.85, 0.86, 0.88, 1.0)
_BAND_COLOR = (0.95, 0.55, 0.10, 1.0)

_BAND_PROUD = 0.0004


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def build_band_housing(
    scene: ManiSkillScene,
    *,
    bore_half_width: float,
    outer_half_width: float,
    wall_height: float,
    floor_half_thickness: float,
    color: tuple[float, float, float, float] = _HOUSING_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    s, r_out, hw, tf = bore_half_width, outer_half_width, wall_height, floor_half_thickness
    if not s < r_out:
        raise ValueError(f"bore_half_width={s} must be smaller than outer_half_width={r_out}")
    mat = _material(color)

    builder = scene.create_actor_builder()
    floor_pose = sapien.Pose(p=[0.0, 0.0, -tf])
    builder.add_box_collision(pose=floor_pose, half_size=[r_out, r_out, tf])
    builder.add_box_visual(pose=floor_pose, half_size=[r_out, r_out, tf], material=mat)

    thickness = (r_out - s) * 0.5
    offset = thickness + s
    walls = [
        ((thickness, r_out, hw * 0.5), [offset, 0.0, hw * 0.5]),
        ((thickness, r_out, hw * 0.5), [-offset, 0.0, hw * 0.5]),
        ((r_out, thickness, hw * 0.5), [0.0, offset, hw * 0.5]),
        ((r_out, thickness, hw * 0.5), [0.0, -offset, hw * 0.5]),
    ]
    for half_size, p in walls:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=half_size)
        builder.add_box_visual(pose=pose, half_size=half_size, material=mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_band_plunger(
    scene: ManiSkillScene,
    *,
    half_width: float,
    half_height: float,
    face_half_height: float,
    band_dz: float,
    band_half_height: float,
    body_color: tuple[float, float, float, float] = _PLUNGER_BODY_COLOR,
    face_color: tuple[float, float, float, float] = _PLUNGER_FACE_COLOR,
    band_color: tuple[float, float, float, float] = _BAND_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    aw, ah, fh = half_width, half_height, face_half_height
    if not 0.0 < fh < ah:
        raise ValueError(f"face_half_height={fh} must be inside the plunger half-height {ah}")
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[aw, aw, ah])

    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, ah - fh]),
        half_size=[aw, aw, fh],
        material=_material(face_color),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, -fh]),
        half_size=[aw, aw, ah - fh],
        material=_material(body_color),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, band_dz]),
        half_size=[aw + _BAND_PROUD, aw + _BAND_PROUD, band_half_height],
        material=_material(band_color),
    )
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
