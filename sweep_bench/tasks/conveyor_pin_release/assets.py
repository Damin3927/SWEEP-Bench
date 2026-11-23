from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_BODY_COLOR = (0.70, 0.68, 0.64, 1.0)
_JOURNAL_COLOR = (0.34, 0.36, 0.41, 1.0)
_FLAP_COLOR = (0.92, 0.45, 0.16, 1.0)
_TAIL_COLOR = (0.55, 0.27, 0.10, 1.0)
_SHANK_COLOR = (0.62, 0.64, 0.67, 1.0)
_POST_COLOR = (0.16, 0.18, 0.22, 1.0)

_X_TO_Z = sapien.Pose(q=[math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0])


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


def _add_square_bore(
    builder: ActorBuilder,
    *,
    axis_x: tuple[float, float],
    axis_y: float,
    axis_z: float,
    clear: float,
    wall_half_thickness: float,
    floor_z: float,
    color: tuple[float, float, float, float],
    density: float,
    material: sapien.physx.PhysxMaterial,
) -> None:
    lo, hi = axis_x
    half_x = 0.5 * (hi - lo)
    mid_x = 0.5 * (hi + lo)
    span_half_y = clear + 2.0 * wall_half_thickness
    ledge_half_z = 0.5 * (axis_z - clear - floor_z)
    _add_box(
        builder,
        (mid_x, axis_y, floor_z + ledge_half_z),
        (half_x, span_half_y, ledge_half_z),
        color,
        density,
        material,
    )
    for sy in (-1.0, 1.0):
        _add_box(
            builder,
            (mid_x, axis_y + sy * (clear + wall_half_thickness), axis_z),
            (half_x, wall_half_thickness, clear),
            color,
            density,
            material,
        )
    _add_box(
        builder,
        (mid_x, axis_y, axis_z + clear + wall_half_thickness),
        (half_x, span_half_y, wall_half_thickness),
        color,
        density,
        material,
    )


def build_pinned_carrier(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    floor_half_thickness: float,
    hinge_y: float,
    hinge_z: float,
    hinge_pin_radius: float,
    hinge_clearance: float,
    journal_inner_x: float,
    journal_outer_x: float,
    wall_half_thickness: float,
    bore_y: float,
    bore_z: float,
    bore_clear: float,
    keeper_x_lo: float,
    keeper_x_hi: float,
    density: float,
    bearing_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not journal_inner_x < journal_outer_x:
        raise ValueError(f"journal x span must be increasing, got [{journal_inner_x}, {journal_outer_x}]")
    if not keeper_x_lo < keeper_x_hi:
        raise ValueError(f"keeper x span must be increasing, got [{keeper_x_lo}, {keeper_x_hi}]")
    floor_top = 2.0 * floor_half_thickness
    if bore_z - bore_clear <= floor_top:
        raise ValueError(f"keeper bore (axis {bore_z}, clear {bore_clear}) must sit above the floor top {floor_top}")

    builder = scene.create_actor_builder()
    bearing = bearing_material(bearing_friction)

    _add_box(builder, (0.0, 0.0, floor_half_thickness), (half_x, half_y, floor_half_thickness), _BODY_COLOR, density)

    hinge_clear = hinge_pin_radius + hinge_clearance
    for sx in (-1.0, 1.0):
        lo = sx * journal_inner_x
        hi = sx * journal_outer_x
        _add_square_bore(
            builder,
            axis_x=(min(lo, hi), max(lo, hi)),
            axis_y=hinge_y,
            axis_z=hinge_z,
            clear=hinge_clear,
            wall_half_thickness=wall_half_thickness,
            floor_z=floor_top,
            color=_JOURNAL_COLOR,
            density=density,
            material=bearing,
        )

    _add_square_bore(
        builder,
        axis_x=(keeper_x_lo, keeper_x_hi),
        axis_y=bore_y,
        axis_z=bore_z,
        clear=bore_clear,
        wall_half_thickness=wall_half_thickness,
        floor_z=floor_top,
        color=_JOURNAL_COLOR,
        density=density,
        material=bearing,
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_spring_flap(
    scene: ManiSkillScene,
    *,
    plate_half_x: float,
    plate_half_y: float,
    plate_half_thickness: float,
    plate_root_gap: float,
    tail_x_lo: float,
    tail_x_hi: float,
    tail_y_near: float,
    tail_y_far: float,
    tail_half_thickness: float,
    tail_underside: float,
    hinge_pin_radius: float,
    hinge_pin_half_length: float,
    collar_radius: float,
    collar_half_length: float,
    collar_x: float,
    density: float,
    bearing_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if collar_radius <= hinge_pin_radius:
        raise ValueError(f"collar_radius ({collar_radius}) must exceed hinge_pin_radius ({hinge_pin_radius})")
    if not (tail_y_far < tail_y_near < 0.0):
        raise ValueError(f"tail must sit behind the hinge: need {tail_y_far} < {tail_y_near} < 0")

    builder = scene.create_actor_builder()
    bearing = bearing_material(bearing_friction)

    builder.add_cylinder_collision(
        radius=hinge_pin_radius, half_length=hinge_pin_half_length, density=density, material=bearing
    )
    builder.add_cylinder_visual(
        radius=hinge_pin_radius, half_length=hinge_pin_half_length, material=_material(_SHANK_COLOR)
    )

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
        material=_material(_FLAP_COLOR),
    )

    tail_half_x = 0.5 * (tail_x_hi - tail_x_lo)
    tail_half_y = 0.5 * (tail_y_near - tail_y_far)
    tail_pose = sapien.Pose(
        p=[
            0.5 * (tail_x_hi + tail_x_lo),
            0.5 * (tail_y_near + tail_y_far),
            tail_underside + tail_half_thickness,
        ]
    )
    builder.add_box_collision(
        pose=tail_pose,
        half_size=[tail_half_x, tail_half_y, tail_half_thickness],
        density=density,
        material=bearing,
    )
    builder.add_box_visual(
        pose=tail_pose,
        half_size=[tail_half_x, tail_half_y, tail_half_thickness],
        material=_material(_TAIL_COLOR),
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
            pose=collar_pose, radius=collar_radius, half_length=collar_half_length, material=_material(_SHANK_COLOR)
        )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_retaining_pin(
    scene: ManiSkillScene,
    *,
    shank_radius: float,
    shank_length: float,
    shank_dz: float,
    post_radius: float,
    post_half_height: float,
    density: float,
    grip_friction: float,
    bore_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if shank_dz >= 0.0:
        raise ValueError(f"shank_dz must be negative (the post stands above the bore), got {shank_dz}")

    builder = scene.create_actor_builder()

    shank_pose = sapien.Pose(p=[0.5 * shank_length, 0.0, shank_dz])
    builder.add_cylinder_collision(
        pose=shank_pose,
        radius=shank_radius,
        half_length=0.5 * shank_length,
        density=density,
        material=bearing_material(bore_friction),
    )
    builder.add_cylinder_visual(
        pose=shank_pose, radius=shank_radius, half_length=0.5 * shank_length, material=_material(_SHANK_COLOR)
    )

    grip_mat = sapien.physx.PhysxMaterial(
        static_friction=grip_friction, dynamic_friction=grip_friction, restitution=0.0
    )
    builder.add_cylinder_collision(
        pose=_X_TO_Z, radius=post_radius, half_length=post_half_height, density=density, material=grip_mat
    )
    builder.add_cylinder_visual(
        pose=_X_TO_Z, radius=post_radius, half_length=post_half_height, material=_material(_POST_COLOR)
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
