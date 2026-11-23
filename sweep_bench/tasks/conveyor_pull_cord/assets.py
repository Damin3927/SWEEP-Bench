from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_BODY_COLOR = (0.30, 0.33, 0.38, 1.0)
_LIP_COLOR = (0.20, 0.22, 0.26, 1.0)
_SHAFT_COLOR = (0.72, 0.74, 0.78, 1.0)
_BAR_COLOR = (0.90, 0.55, 0.12, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def _box(
    builder: ActorBuilder,
    center: tuple[float, float, float],
    half: tuple[float, float, float],
    color: tuple[float, float, float, float],
    density: float,
) -> None:
    pose = sapien.Pose(p=center)
    builder.add_box_collision(pose=pose, half_size=half, density=density)
    builder.add_box_visual(pose=pose, half_size=half, material=_material(color))


def build_cord_body(
    scene: ManiSkillScene,
    *,
    base_half_xy: float,
    base_half_thickness: float,
    tube_outer_half: float,
    bore_half: float,
    tube_height: float,
    lip_opening_half: float,
    lip_thickness: float,
    density: float,
) -> ActorBuilder:
    if not lip_opening_half < bore_half < tube_outer_half <= base_half_xy:
        raise ValueError(
            f"bad radial stack: lip_opening_half={lip_opening_half} bore_half={bore_half} "
            f"tube_outer_half={tube_outer_half} base_half_xy={base_half_xy}"
        )

    builder = scene.create_actor_builder()

    _box(
        builder,
        (0.0, 0.0, -base_half_thickness),
        (base_half_xy, base_half_xy, base_half_thickness),
        _BODY_COLOR,
        density,
    )

    wall_half = 0.5 * (tube_outer_half - bore_half)
    wall_offset = bore_half + wall_half
    half_h = 0.5 * tube_height
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        half = (wall_half, tube_outer_half, half_h) if sx else (tube_outer_half, wall_half, half_h)
        _box(builder, (sx * wall_offset, sy * wall_offset, half_h), half, _BODY_COLOR, density)

    lip_half = 0.5 * (tube_outer_half - lip_opening_half)
    lip_offset = lip_opening_half + lip_half
    lip_cz = tube_height + 0.5 * lip_thickness
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        half = (
            (lip_half, tube_outer_half, 0.5 * lip_thickness) if sx else (tube_outer_half, lip_half, 0.5 * lip_thickness)
        )
        _box(builder, (sx * lip_offset, sy * lip_offset, lip_cz), half, _LIP_COLOR, density)

    return builder


def build_pull_handle(
    scene: ManiSkillScene,
    *,
    collar_half: float,
    collar_half_thickness: float,
    shaft_half: float,
    bar_half_x: float,
    bar_half_y: float,
    bar_half_z: float,
    rest_local_z: float,
    density: float,
) -> ActorBuilder:
    if not shaft_half < collar_half:
        raise ValueError(f"shaft_half={shaft_half} must be under collar_half={collar_half}")

    builder = scene.create_actor_builder()

    def local(z: float) -> float:
        return z - rest_local_z

    collar_cz = collar_half_thickness
    shaft_lo = 2.0 * collar_half_thickness
    shaft_hi = rest_local_z - bar_half_z
    if not shaft_lo < shaft_hi:
        raise ValueError(f"rest_local_z={rest_local_z} leaves no room for the shaft")

    _box(
        builder, (0.0, 0.0, local(collar_cz)), (collar_half, collar_half, collar_half_thickness), _SHAFT_COLOR, density
    )
    _box(
        builder,
        (0.0, 0.0, local(0.5 * (shaft_lo + shaft_hi))),
        (shaft_half, shaft_half, 0.5 * (shaft_hi - shaft_lo)),
        _SHAFT_COLOR,
        density,
    )
    _box(builder, (0.0, 0.0, 0.0), (bar_half_x, bar_half_y, bar_half_z), _BAR_COLOR, density)

    return builder
