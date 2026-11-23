from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_BODY_COLOR = (0.72, 0.70, 0.66, 1.0)
_JOURNAL_COLOR = (0.35, 0.37, 0.42, 1.0)
_LID_COLOR = (0.90, 0.40, 0.22, 1.0)
_RIB_COLOR = (0.20, 0.22, 0.26, 1.0)
_PIN_COLOR = (0.55, 0.57, 0.60, 1.0)


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
) -> None:
    pose = sapien.Pose(p=list(center))
    builder.add_box_collision(pose=pose, half_size=half, density=density, material=material)
    builder.add_box_visual(pose=pose, half_size=half, material=_material(color))


def build_hinged_lid_box(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    floor_half_thickness: float,
    rim_z: float,
    wall_half_thickness: float,
    pin_y: float,
    pin_z: float,
    pin_radius: float,
    pin_clearance: float,
    journal_inner_x: float,
    journal_outer_x: float,
    post_inner_x: float,
    post_outer_x: float,
    post_half_y: float,
    post_stop_dy: float,
    post_top_z: float,
    rail_y_lo: float,
    density: float,
    bearing_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if rim_z >= pin_z:
        raise ValueError(f"rim_z ({rim_z}) must be below the pin axis ({pin_z}) for the lid to rest on it")
    if not journal_inner_x < journal_outer_x:
        raise ValueError(f"journal x span must be increasing, got [{journal_inner_x}, {journal_outer_x}]")
    if not post_outer_x <= journal_inner_x:
        raise ValueError(f"backstop posts ({post_outer_x}) must sit inboard of the journals ({journal_inner_x})")

    builder = scene.create_actor_builder()

    _add_box(builder, (0.0, 0.0, floor_half_thickness), (half_x, half_y, floor_half_thickness), _BODY_COLOR, density)

    floor_top = 2.0 * floor_half_thickness
    wall_half_z = 0.5 * (rim_z - floor_top)
    wall_mid_z = floor_top + wall_half_z

    _add_box(
        builder,
        (0.0, half_y - wall_half_thickness, wall_mid_z),
        (half_x, wall_half_thickness, wall_half_z),
        _BODY_COLOR,
        density,
    )
    if rail_y_lo >= half_y:
        raise ValueError(f"rail_y_lo ({rail_y_lo}) must be inside the footprint (half_y {half_y})")
    rail_half_y = 0.5 * (half_y - rail_y_lo)
    for sx in (-1.0, 1.0):
        _add_box(
            builder,
            (sx * (half_x - wall_half_thickness), rail_y_lo + rail_half_y, wall_mid_z),
            (wall_half_thickness, rail_half_y, wall_half_z),
            _BODY_COLOR,
            density,
        )

    bearing = bearing_material(bearing_friction)
    clear = pin_radius + pin_clearance
    journal_half_x = 0.5 * (journal_outer_x - journal_inner_x)
    journal_mid_x = 0.5 * (journal_outer_x + journal_inner_x)
    pocket_half_y = clear + 2.0 * wall_half_thickness
    cap_half_z = 0.5 * wall_half_thickness
    for sx in (-1.0, 1.0):
        jx = sx * journal_mid_x
        ledge_half_z = 0.5 * (pin_z - clear - floor_top)
        _add_box(
            builder,
            (jx, pin_y, floor_top + ledge_half_z),
            (journal_half_x, pocket_half_y, ledge_half_z),
            _JOURNAL_COLOR,
            density,
            bearing,
        )
        for sy in (-1.0, 1.0):
            _add_box(
                builder,
                (jx, pin_y + sy * (clear + wall_half_thickness), pin_z),
                (journal_half_x, wall_half_thickness, clear),
                _JOURNAL_COLOR,
                density,
                bearing,
            )
        _add_box(
            builder,
            (jx, pin_y, pin_z + clear + cap_half_z),
            (journal_half_x, pocket_half_y, cap_half_z),
            _JOURNAL_COLOR,
            density,
            bearing,
        )

    post_half_x = 0.5 * (post_outer_x - post_inner_x)
    post_mid_x = 0.5 * (post_outer_x + post_inner_x)
    post_half_z = 0.5 * (post_top_z - floor_top)
    for sx in (-1.0, 1.0):
        _add_box(
            builder,
            (sx * post_mid_x, pin_y - post_stop_dy - post_half_y, floor_top + post_half_z),
            (post_half_x, post_half_y, post_half_z),
            _JOURNAL_COLOR,
            density,
        )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_hinged_lid(
    scene: ManiSkillScene,
    *,
    plate_half_x: float,
    plate_half_y: float,
    plate_half_thickness: float,
    plate_root_gap: float,
    rib_half_y: float,
    rib_half_height: float,
    rib_centre_y: float,
    pin_radius: float,
    pin_half_length: float,
    collar_radius: float,
    collar_half_length: float,
    collar_x: float,
    density: float,
    bearing_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if collar_radius <= pin_radius:
        raise ValueError(f"collar_radius ({collar_radius}) must exceed pin_radius ({pin_radius}) to act as a stop")

    builder = scene.create_actor_builder()
    bearing = bearing_material(bearing_friction)

    builder.add_cylinder_collision(radius=pin_radius, half_length=pin_half_length, density=density, material=bearing)
    builder.add_cylinder_visual(radius=pin_radius, half_length=pin_half_length, material=_material(_PIN_COLOR))

    plate_pose = sapien.Pose(p=[0.0, plate_root_gap + plate_half_y, 0.0])
    builder.add_box_collision(
        pose=plate_pose,
        half_size=[plate_half_x, plate_half_y, plate_half_thickness],
        density=density,
        material=bearing,
    )
    builder.add_box_visual(
        pose=plate_pose,
        half_size=[plate_half_x, plate_half_y, plate_half_thickness],
        material=_material(_LID_COLOR),
    )

    rib_pose = sapien.Pose(p=[0.0, rib_centre_y, plate_half_thickness + rib_half_height])
    builder.add_box_collision(
        pose=rib_pose,
        half_size=[plate_half_x, rib_half_y, rib_half_height],
        density=density,
        material=bearing,
    )
    builder.add_box_visual(
        pose=rib_pose,
        half_size=[plate_half_x, rib_half_y, rib_half_height],
        material=_material(_RIB_COLOR),
    )

    for sx in (-1.0, 1.0):
        collar_pose = sapien.Pose(p=[sx * collar_x, 0.0, 0.0])
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


def stop_corner_offset(stop_angle_deg: float, radius: float) -> tuple[float, float]:
    theta = math.radians(stop_angle_deg)
    return radius * math.cos(theta), radius * math.sin(theta)
