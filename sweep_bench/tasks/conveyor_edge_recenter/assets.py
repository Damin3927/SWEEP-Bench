from __future__ import annotations

from typing import TYPE_CHECKING

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder


def build_parcel(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_z: float,
    color: tuple[float, float, float, float],
    density: float,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    half = [half_x, half_y, half_z]
    builder.add_box_collision(half_size=half, density=density)
    builder.add_box_visual(
        half_size=half,
        material=flat(color, roughness=0.7),
    )
    return builder
