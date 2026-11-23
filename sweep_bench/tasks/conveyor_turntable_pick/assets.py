from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_UPRIGHT = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])

_PLINTH_COLOR = (0.30, 0.32, 0.36, 1.0)
_BOSS_COLOR = (0.42, 0.44, 0.48, 1.0)
_TABLE_COLOR = (0.86, 0.85, 0.80, 1.0)
_POCKET_COLOR = (0.34, 0.36, 0.40, 1.0)
_PIN_COLOR = (0.60, 0.62, 0.66, 1.0)
_PART_COLOR = (0.88, 0.42, 0.14, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def journal_material(friction: float) -> sapien.physx.PhysxMaterial:
    return sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)


def _add_box(
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


def _add_square_ring(
    builder: ActorBuilder,
    inner_half: float,
    outer_half: float,
    z_lo: float,
    z_hi: float,
    color: tuple[float, float, float, float],
    density: float,
    material: sapien.physx.PhysxMaterial | None = None,
) -> None:
    half_z = 0.5 * (z_hi - z_lo)
    mid_z = 0.5 * (z_lo + z_hi)
    span = 0.5 * (outer_half - inner_half)
    mid = 0.5 * (outer_half + inner_half)
    for sx in (-1.0, 1.0):
        _add_box(builder, (sx * mid, 0.0, mid_z), (span, outer_half, half_z), color, density, material)
    for sy in (-1.0, 1.0):
        _add_box(builder, (0.0, sy * mid, mid_z), (inner_half, span, half_z), color, density, material)


def build_turntable_plinth(
    scene: ManiSkillScene,
    *,
    plinth_radius: float,
    plinth_half_thickness: float,
    boss_half_width: float,
    well_half_width: float,
    well_depth: float,
    ledge_thickness: float,
    aperture_half_width: float,
    thrust_pad_thickness: float,
    density: float,
    journal_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not aperture_half_width < well_half_width < boss_half_width < plinth_radius:
        raise ValueError(
            "plinth radii must increase: aperture "
            f"({aperture_half_width}) < well ({well_half_width}) < boss "
            f"({boss_half_width}) < plinth ({plinth_radius})"
        )

    builder = scene.create_actor_builder()
    journal = journal_material(journal_friction)

    slab_pose = sapien.Pose(p=[0.0, 0.0, plinth_half_thickness], q=_UPRIGHT.q)
    builder.add_cylinder_collision(
        pose=slab_pose, radius=plinth_radius, half_length=plinth_half_thickness, density=density
    )
    builder.add_cylinder_visual(
        pose=slab_pose, radius=plinth_radius, half_length=plinth_half_thickness, material=_material(_PLINTH_COLOR)
    )

    slab_top = 2.0 * plinth_half_thickness
    _add_box(
        builder,
        (0.0, 0.0, slab_top + 0.5 * thrust_pad_thickness),
        (well_half_width, well_half_width, 0.5 * thrust_pad_thickness),
        _PIN_COLOR,
        density,
        journal,
    )

    well_floor = slab_top + thrust_pad_thickness
    ledge_lo = well_floor + well_depth

    _add_square_ring(builder, well_half_width, boss_half_width, slab_top, ledge_lo, _BOSS_COLOR, density, journal)
    _add_square_ring(
        builder,
        aperture_half_width,
        boss_half_width,
        ledge_lo,
        ledge_lo + ledge_thickness,
        _BOSS_COLOR,
        density,
        journal,
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_turntable(
    scene: ManiSkillScene,
    *,
    collar_radius: float,
    collar_half_thickness: float,
    pin_radius: float,
    pin_half_length: float,
    disc_radius: float,
    disc_half_thickness: float,
    plate_bottom_z: float,
    seat_radius: float,
    pocket_half_x: float,
    pocket_half_y: float,
    wall_thickness: float,
    wall_height: float,
    density: float,
    friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if collar_radius <= pin_radius:
        raise ValueError(f"collar_radius ({collar_radius}) must exceed pin_radius ({pin_radius}) to act as a stop")
    if plate_bottom_z <= collar_half_thickness + 2.0 * pin_half_length - 1e-9:
        raise ValueError(
            f"plate_bottom_z ({plate_bottom_z}) must sit at or above the pin's top "
            f"({collar_half_thickness + 2.0 * pin_half_length})"
        )
    corner = math.hypot(seat_radius + pocket_half_x + wall_thickness, pocket_half_y + wall_thickness)
    if corner > disc_radius:
        raise ValueError(f"seat pocket corner at r={corner:.4f} overhangs the {disc_radius:.4f} m table")

    builder = scene.create_actor_builder()
    journal = journal_material(friction)

    builder.add_cylinder_collision(
        pose=_UPRIGHT, radius=collar_radius, half_length=collar_half_thickness, density=density, material=journal
    )
    builder.add_cylinder_visual(
        pose=_UPRIGHT, radius=collar_radius, half_length=collar_half_thickness, material=_material(_PIN_COLOR)
    )

    pin_pose = sapien.Pose(p=[0.0, 0.0, collar_half_thickness + pin_half_length], q=_UPRIGHT.q)
    builder.add_cylinder_collision(
        pose=pin_pose, radius=pin_radius, half_length=pin_half_length, density=density, material=journal
    )
    builder.add_cylinder_visual(
        pose=pin_pose, radius=pin_radius, half_length=pin_half_length, material=_material(_PIN_COLOR)
    )

    plate_pose = sapien.Pose(p=[0.0, 0.0, plate_bottom_z + disc_half_thickness], q=_UPRIGHT.q)
    builder.add_cylinder_collision(
        pose=plate_pose, radius=disc_radius, half_length=disc_half_thickness, density=density
    )
    builder.add_cylinder_visual(
        pose=plate_pose, radius=disc_radius, half_length=disc_half_thickness, material=_material(_TABLE_COLOR)
    )

    plate_top = plate_bottom_z + 2.0 * disc_half_thickness
    mid_z = plate_top + 0.5 * wall_height
    half_z = 0.5 * wall_height
    outer_y = pocket_half_y + wall_thickness
    for sx in (-1.0, 1.0):
        _add_box(
            builder,
            (seat_radius + sx * (pocket_half_x + 0.5 * wall_thickness), 0.0, mid_z),
            (0.5 * wall_thickness, outer_y, half_z),
            _POCKET_COLOR,
            density,
        )
    for sy in (-1.0, 1.0):
        _add_box(
            builder,
            (seat_radius, sy * (pocket_half_y + 0.5 * wall_thickness), mid_z),
            (pocket_half_x, 0.5 * wall_thickness, half_z),
            _POCKET_COLOR,
            density,
        )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_seated_part(
    scene: ManiSkillScene,
    *,
    half_width: float,
    half_length: float,
    half_height: float,
    density: float,
    friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    material = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    _add_box(builder, (0.0, 0.0, 0.0), (half_width, half_length, half_height), _PART_COLOR, density, material)
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
