from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_DECK_COLOR = (0.40, 0.42, 0.46, 1.0)
_RAIL_COLOR = (0.92, 0.50, 0.12, 1.0)

CARRIER_FRICTION = 0.5


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6, metallic=0.1)


def build_rail_carrier(
    scene: ManiSkillScene,
    *,
    plate_half_x: float,
    plate_half_y: float,
    plate_half_thickness: float,
    rail_half_width: float,
    rail_half_length: float,
    rail_height: float,
    friction: float = CARRIER_FRICTION,
    deck_color: tuple[float, float, float, float] = _DECK_COLOR,
    rail_color: tuple[float, float, float, float] = _RAIL_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not rail_half_width < plate_half_x:
        raise ValueError(f"rail_half_width={rail_half_width} must be smaller than plate_half_x={plate_half_x}")
    if not rail_half_length <= plate_half_y:
        raise ValueError(f"rail_half_length={rail_half_length} must fit inside plate_half_y={plate_half_y}")
    if rail_height <= 0.0:
        raise ValueError(f"rail_height={rail_height} must be positive")

    phys = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    builder = scene.create_actor_builder()

    deck_pose = sapien.Pose(p=[0.0, 0.0, -plate_half_thickness])
    deck_half = [plate_half_x, plate_half_y, plate_half_thickness]
    builder.add_box_collision(pose=deck_pose, half_size=deck_half, material=phys)
    builder.add_box_visual(pose=deck_pose, half_size=deck_half, material=_material(deck_color))

    rail_pose = sapien.Pose(p=[0.0, 0.0, rail_height * 0.5])
    rail_half = [rail_half_width, rail_half_length, rail_height * 0.5]
    builder.add_box_collision(pose=rail_pose, half_size=rail_half, material=phys)
    builder.add_box_visual(pose=rail_pose, half_size=rail_half, material=_material(rail_color))

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
