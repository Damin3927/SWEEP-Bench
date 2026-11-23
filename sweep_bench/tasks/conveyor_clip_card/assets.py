from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_CARRIER_COLOR = (0.70, 0.68, 0.64, 1.0)
_JOURNAL_COLOR = (0.34, 0.36, 0.41, 1.0)
_PAD_COLOR = (0.90, 0.38, 0.20, 1.0)
_PIN_COLOR = (0.55, 0.57, 0.60, 1.0)
_SHIM_COLOR = (0.18, 0.45, 0.80, 1.0)
_TAB_COLOR = (0.12, 0.24, 0.45, 1.0)

_AXIS_Y_Q = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def bearing_material(friction: float) -> sapien.physx.PhysxMaterial:
    return sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)


def _add_box(
    builder: ActorBuilder,
    center: tuple[float, float, float],
    half: tuple[float, float, float],
    color: tuple[float, float, float, float],
    density: float,
    material: sapien.physx.PhysxMaterial | None = None,
    q: list[float] | None = None,
) -> None:
    pose = sapien.Pose(p=list(center)) if q is None else sapien.Pose(p=list(center), q=q)
    builder.add_box_collision(pose=pose, half_size=half, density=density, material=material)
    builder.add_box_visual(pose=pose, half_size=half, material=_material(color))


def build_clip_carrier(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    deck_half_thickness: float,
    pin_x: float,
    pin_height: float,
    pin_radius: float,
    pin_clearance: float,
    journal_inner_y: float,
    journal_outer_y: float,
    wall_half_thickness: float,
    density: float,
    bearing_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    deck_z = 2.0 * deck_half_thickness
    clear = pin_radius + pin_clearance
    if pin_height <= clear:
        raise ValueError(f"pin_height ({pin_height}) must exceed the pocket clearance ({clear}) to clear the deck")
    if not journal_inner_y < journal_outer_y:
        raise ValueError(f"journal y span must be increasing, got [{journal_inner_y}, {journal_outer_y}]")
    if journal_outer_y >= half_y:
        raise ValueError(f"journal_outer_y ({journal_outer_y}) must sit inside the footprint (half_y {half_y})")

    builder = scene.create_actor_builder()

    _add_box(builder, (0.0, 0.0, deck_half_thickness), (half_x, half_y, deck_half_thickness), _CARRIER_COLOR, density)

    bearing = bearing_material(bearing_friction)
    pin_z = deck_z + pin_height
    journal_half_y = 0.5 * (journal_outer_y - journal_inner_y)
    journal_mid_y = 0.5 * (journal_outer_y + journal_inner_y)
    pocket_half_x = clear + 2.0 * wall_half_thickness
    cap_half_z = 0.5 * wall_half_thickness
    for sy in (-1.0, 1.0):
        jy = sy * journal_mid_y
        ledge_half_z = 0.5 * (pin_z - clear - deck_z)
        _add_box(
            builder,
            (pin_x, jy, deck_z + ledge_half_z),
            (pocket_half_x, journal_half_y, ledge_half_z),
            _JOURNAL_COLOR,
            density,
            bearing,
        )
        for sx in (-1.0, 1.0):
            _add_box(
                builder,
                (pin_x + sx * (clear + wall_half_thickness), jy, pin_z),
                (wall_half_thickness, journal_half_y, clear),
                _JOURNAL_COLOR,
                density,
                bearing,
            )
        _add_box(
            builder,
            (pin_x, jy, pin_z + clear + cap_half_z),
            (pocket_half_x, journal_half_y, cap_half_z),
            _JOURNAL_COLOR,
            density,
            bearing,
        )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_clip_pad(
    scene: ManiSkillScene,
    *,
    pad_length: float,
    plate_half_y: float,
    plate_half_thickness: float,
    plate_root_gap: float,
    pin_height: float,
    lip_half_length: float,
    lip_angle: float,
    pin_radius: float,
    pin_half_length: float,
    collar_radius: float,
    collar_half_length: float,
    collar_y: float,
    density: float,
    bearing_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if collar_radius <= pin_radius:
        raise ValueError(f"collar_radius ({collar_radius}) must exceed pin_radius ({pin_radius}) to act as a stop")
    if pad_length <= plate_root_gap:
        raise ValueError(f"pad_length ({pad_length}) must exceed the root gap ({plate_root_gap})")

    builder = scene.create_actor_builder()
    bearing = bearing_material(bearing_friction)

    builder.add_cylinder_collision(
        pose=sapien.Pose(q=_AXIS_Y_Q),
        radius=pin_radius,
        half_length=pin_half_length,
        density=density,
        material=bearing,
    )
    builder.add_cylinder_visual(
        pose=sapien.Pose(q=_AXIS_Y_Q),
        radius=pin_radius,
        half_length=pin_half_length,
        material=_material(_PIN_COLOR),
    )

    plate_bottom = -pin_height
    plate_half_x = 0.5 * (pad_length - plate_root_gap)
    _add_box(
        builder,
        (-(plate_root_gap + plate_half_x), 0.0, plate_bottom + plate_half_thickness),
        (plate_half_x, plate_half_y, plate_half_thickness),
        _PAD_COLOR,
        density,
    )

    knee = (-pad_length, plate_bottom)
    cos_a, sin_a = math.cos(lip_angle), math.sin(lip_angle)
    lip_center = (
        knee[0] - lip_half_length * cos_a + plate_half_thickness * sin_a,
        0.0,
        knee[1] + lip_half_length * sin_a + plate_half_thickness * cos_a,
    )
    _add_box(
        builder,
        lip_center,
        (lip_half_length, plate_half_y, plate_half_thickness),
        _PAD_COLOR,
        density,
        q=[math.cos(0.5 * lip_angle), 0.0, math.sin(0.5 * lip_angle), 0.0],
    )

    for sy in (-1.0, 1.0):
        collar_pose = sapien.Pose(p=[0.0, sy * collar_y, 0.0], q=_AXIS_Y_Q)
        builder.add_cylinder_collision(
            pose=collar_pose,
            radius=collar_radius,
            half_length=collar_half_length,
            density=density,
            material=bearing,
        )
        builder.add_cylinder_visual(
            pose=collar_pose,
            radius=collar_radius,
            half_length=collar_half_length,
            material=_material(_PIN_COLOR),
        )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_shim(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_thickness: float,
    tab_half_x: float,
    tab_half_y: float,
    tab_half_height: float,
    tab_center_x: float,
    density: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    _add_box(builder, (0.0, 0.0, 0.0), (half_x, half_y, half_thickness), _SHIM_COLOR, density)
    _add_box(
        builder,
        (tab_center_x, 0.0, half_thickness + tab_half_height),
        (tab_half_x, tab_half_y, tab_half_height),
        _TAB_COLOR,
        density,
    )
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
