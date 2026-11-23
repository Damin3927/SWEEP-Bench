from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_BODY_COLOR = (0.86, 0.78, 0.55, 1.0)
_BASE_COLOR = (0.22, 0.27, 0.40, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.7)


def build_standing_block(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_z: float,
    density: float,
    base_band_frac: float = 0.25,
    body_color: tuple[float, float, float, float] = _BODY_COLOR,
    base_color: tuple[float, float, float, float] = _BASE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not 0.0 < base_band_frac < 1.0:
        raise ValueError(f"base_band_frac must be in (0, 1), got {base_band_frac}")

    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[half_x, half_y, half_z], density=density)

    band_half = half_z * base_band_frac
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, -half_z + band_half]),
        half_size=[half_x, half_y, band_half],
        material=_material(base_color),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, band_half]),
        half_size=[half_x, half_y, half_z - band_half],
        material=_material(body_color),
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
