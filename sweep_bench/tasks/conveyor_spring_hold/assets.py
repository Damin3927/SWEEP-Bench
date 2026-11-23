from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.physx
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_CARRIER_COLOR = (0.26, 0.30, 0.38, 1.0)
_RAIL_COLOR = (0.18, 0.21, 0.27, 1.0)
_TAB_COLOR = (0.88, 0.82, 0.55, 1.0)
_BAND_COLOR = (0.85, 0.25, 0.18, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.7)


def _box(
    builder: ActorBuilder,
    center: tuple[float, float, float],
    half: tuple[float, float, float],
    color: tuple[float, float, float, float],
    density: float,
    material: sapien.physx.PhysxMaterial | None = None,
) -> None:
    pose = sapien.Pose(p=list(center))
    builder.add_box_collision(pose=pose, half_size=half, density=density, material=material)
    builder.add_box_visual(pose=pose, half_size=half, material=_material(color))


def build_spring_tab_carrier(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    plate_half_z: float,
    rail_inner_y: float,
    rail_outer_y: float,
    rail_height: float,
    stop_inner_x: float,
    stop_thickness: float,
    stop_height: float,
    band_x: float,
    band_half_x: float,
    density: float,
) -> ActorBuilder:
    builder = scene.create_actor_builder()

    _box(builder, (0.0, 0.0, 0.0), (half_x, half_y, plate_half_z), _CARRIER_COLOR, density)

    rail_half_y = 0.5 * (rail_outer_y - rail_inner_y)
    rail_centre_y = 0.5 * (rail_outer_y + rail_inner_y)
    rail_centre_z = plate_half_z + 0.5 * rail_height
    for sign in (-1.0, 1.0):
        _box(
            builder,
            (0.0, sign * rail_centre_y, rail_centre_z),
            (half_x, rail_half_y, 0.5 * rail_height),
            _RAIL_COLOR,
            density,
        )

    _box(
        builder,
        (stop_inner_x - 0.5 * stop_thickness, 0.0, plate_half_z + 0.5 * stop_height),
        (0.5 * stop_thickness, rail_outer_y, 0.5 * stop_height),
        _RAIL_COLOR,
        density,
    )

    builder.add_box_visual(
        pose=sapien.Pose(p=[band_x, 0.0, plate_half_z + 0.0005]),
        half_size=[band_half_x, rail_inner_y, 0.0005],
        material=_material(_BAND_COLOR),
    )
    return builder


def build_spring_tab(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_z: float,
    density: float,
    friction: float,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    material = sapien.physx.PhysxMaterial(
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
    )
    _box(builder, (0.0, 0.0, 0.0), (half_x, half_y, half_z), _TAB_COLOR, density, material=material)
    return builder
