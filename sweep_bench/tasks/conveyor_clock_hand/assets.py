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

_FACE_COLOR = (0.92, 0.90, 0.84, 1.0)
_BOSS_COLOR = (0.34, 0.36, 0.41, 1.0)
_PLATE_COLOR = (0.16, 0.18, 0.22, 1.0)
_POINTER_COLOR = (0.90, 0.26, 0.18, 1.0)
_TAIL_COLOR = (0.22, 0.28, 0.45, 1.0)
_PIN_COLOR = (0.60, 0.62, 0.66, 1.0)


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
) -> None:
    half_z = 0.5 * (z_hi - z_lo)
    mid_z = 0.5 * (z_lo + z_hi)
    span = 0.5 * (outer_half - inner_half)
    mid = 0.5 * (outer_half + inner_half)
    for sx in (-1.0, 1.0):
        _add_box(builder, (sx * mid, 0.0, mid_z), (span, outer_half, half_z), color, density)
    for sy in (-1.0, 1.0):
        _add_box(builder, (0.0, sy * mid, mid_z), (inner_half, span, half_z), color, density)


def build_clock_carrier(
    scene: ManiSkillScene,
    *,
    disc_radius: float,
    disc_half_thickness: float,
    boss_half_width: float,
    well_half_width: float,
    well_depth: float,
    ledge_thickness: float,
    aperture_half_width: float,
    density: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not aperture_half_width < well_half_width < boss_half_width < disc_radius:
        raise ValueError(
            "clock carrier radii must increase: aperture "
            f"({aperture_half_width}) < well ({well_half_width}) < boss "
            f"({boss_half_width}) < disc ({disc_radius})"
        )

    builder = scene.create_actor_builder()

    face_pose = sapien.Pose(p=[0.0, 0.0, disc_half_thickness], q=_UPRIGHT.q)
    builder.add_cylinder_collision(pose=face_pose, radius=disc_radius, half_length=disc_half_thickness, density=density)
    builder.add_cylinder_visual(
        pose=face_pose, radius=disc_radius, half_length=disc_half_thickness, material=_material(_FACE_COLOR)
    )

    face_top = 2.0 * disc_half_thickness
    ledge_lo = face_top + well_depth

    _add_square_ring(builder, well_half_width, boss_half_width, face_top, ledge_lo, _BOSS_COLOR, density)
    _add_square_ring(
        builder, aperture_half_width, boss_half_width, ledge_lo, ledge_lo + ledge_thickness, _BOSS_COLOR, density
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_clock_hand(
    scene: ManiSkillScene,
    *,
    collar_radius: float,
    collar_half_thickness: float,
    pin_radius: float,
    pin_half_length: float,
    plate_half_width: float,
    plate_half_length: float,
    plate_half_thickness: float,
    pad_radius: float,
    pad_half_width: float,
    pad_half_height: float,
    density: float,
    friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if collar_radius <= pin_radius:
        raise ValueError(f"collar_radius ({collar_radius}) must exceed pin_radius ({pin_radius}) to act as a stop")

    builder = scene.create_actor_builder()
    journal = journal_material(friction)

    builder.add_cylinder_collision(
        pose=_UPRIGHT, radius=collar_radius, half_length=collar_half_thickness, density=density, material=journal
    )
    builder.add_cylinder_visual(
        pose=_UPRIGHT, radius=collar_radius, half_length=collar_half_thickness, material=_material(_PIN_COLOR)
    )

    pin_z = collar_half_thickness + pin_half_length
    pin_pose = sapien.Pose(p=[0.0, 0.0, pin_z], q=_UPRIGHT.q)
    builder.add_cylinder_collision(
        pose=pin_pose, radius=pin_radius, half_length=pin_half_length, density=density, material=journal
    )
    builder.add_cylinder_visual(
        pose=pin_pose, radius=pin_radius, half_length=pin_half_length, material=_material(_PIN_COLOR)
    )

    plate_z = collar_half_thickness + 2.0 * pin_half_length + plate_half_thickness
    _add_box(
        builder,
        (0.0, 0.0, plate_z),
        (plate_half_width, plate_half_length, plate_half_thickness),
        _PLATE_COLOR,
        density,
        journal,
    )

    pad_z = plate_z + plate_half_thickness + pad_half_height
    for sy, color in ((1.0, _POINTER_COLOR), (-1.0, _TAIL_COLOR)):
        _add_box(
            builder,
            (0.0, sy * pad_radius, pad_z),
            (pad_half_width, pad_half_width, pad_half_height),
            color,
            density,
            journal,
        )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
