from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_FRAME_COLOR = (0.70, 0.69, 0.66, 1.0)
_RAIL_COLOR = (0.24, 0.26, 0.30, 1.0)
_JOURNAL_COLOR = (0.36, 0.38, 0.43, 1.0)
_PENDANT_COLOR = (0.86, 0.33, 0.18, 1.0)
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


def build_carry_frame(
    scene: ManiSkillScene,
    *,
    foot_half_x: float,
    foot_half_y: float,
    foot_half_thickness: float,
    post_inner_x: float,
    post_outer_x: float,
    post_inner_y: float,
    post_outer_y: float,
    rail_half_x: float,
    rail_half_y: float,
    rail_half_thickness: float,
    rail_center_z: float,
    pin_z: float,
    pin_radius: float,
    pin_clearance: float,
    journal_inner_x: float,
    journal_outer_x: float,
    wall_half_thickness: float,
    density: float,
    bearing_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    rail_bottom = rail_center_z - rail_half_thickness
    clear = pin_radius + pin_clearance
    if not pin_z + clear + 2.0 * wall_half_thickness < rail_bottom:
        raise ValueError(f"journal cap at {pin_z + clear + 2.0 * wall_half_thickness} does not fit under {rail_bottom}")
    if not journal_inner_x < journal_outer_x:
        raise ValueError(f"journal x span must be increasing, got [{journal_inner_x}, {journal_outer_x}]")
    if not post_inner_x < post_outer_x or not post_inner_y < post_outer_y:
        raise ValueError("post x/y spans must be increasing")

    builder = scene.create_actor_builder()
    foot_top = 2.0 * foot_half_thickness

    _add_box(
        builder,
        (0.0, 0.0, foot_half_thickness),
        (foot_half_x, foot_half_y, foot_half_thickness),
        _FRAME_COLOR,
        density,
    )

    post_half_x = 0.5 * (post_outer_x - post_inner_x)
    post_mid_x = 0.5 * (post_outer_x + post_inner_x)
    post_half_y = 0.5 * (post_outer_y - post_inner_y)
    post_mid_y = 0.5 * (post_outer_y + post_inner_y)
    post_half_z = 0.5 * (rail_bottom - foot_top)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            _add_box(
                builder,
                (sx * post_mid_x, sy * post_mid_y, foot_top + post_half_z),
                (post_half_x, post_half_y, post_half_z),
                _FRAME_COLOR,
                density,
            )

    for sy in (-1.0, 1.0):
        _add_box(
            builder,
            (0.0, sy * post_mid_y, rail_center_z),
            (post_outer_x, post_half_y, rail_half_thickness),
            _FRAME_COLOR,
            density,
        )

    _add_box(
        builder,
        (0.0, 0.0, rail_center_z),
        (rail_half_x, rail_half_y, rail_half_thickness),
        _RAIL_COLOR,
        density,
    )

    bearing = bearing_material(bearing_friction)
    journal_half_x = 0.5 * (journal_outer_x - journal_inner_x)
    journal_mid_x = 0.5 * (journal_outer_x + journal_inner_x)
    pocket_half_y = clear + 2.0 * wall_half_thickness
    for sx in (-1.0, 1.0):
        jx = sx * journal_mid_x
        _add_box(
            builder,
            (jx, 0.0, pin_z - clear - wall_half_thickness),
            (journal_half_x, pocket_half_y, wall_half_thickness),
            _JOURNAL_COLOR,
            density,
            bearing,
        )
        wall_half_z = 0.5 * (rail_bottom - (pin_z - clear))
        for sy in (-1.0, 1.0):
            _add_box(
                builder,
                (jx, sy * (clear + wall_half_thickness), pin_z - clear + wall_half_z),
                (journal_half_x, wall_half_thickness, wall_half_z),
                _JOURNAL_COLOR,
                density,
                bearing,
            )
        _add_box(
            builder,
            (jx, 0.0, pin_z + clear + wall_half_thickness),
            (journal_half_x, pocket_half_y, wall_half_thickness),
            _JOURNAL_COLOR,
            density,
            bearing,
        )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_pendant(
    scene: ManiSkillScene,
    *,
    pin_radius: float,
    pin_half_length: float,
    collar_radius: float,
    collar_half_length: float,
    collar_x: float,
    stem_half_x: float,
    stem_half_y: float,
    stem_half_length: float,
    bob_half: tuple[float, float, float],
    bob_center_z: float,
    stem_density: float,
    bob_density: float,
    bearing_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if collar_radius <= pin_radius:
        raise ValueError(f"collar_radius ({collar_radius}) must exceed pin_radius ({pin_radius}) to act as a stop")
    if bob_center_z >= 0.0:
        raise ValueError(f"bob_center_z ({bob_center_z}) must hang BELOW the pin")

    builder = scene.create_actor_builder()
    bearing = bearing_material(bearing_friction)

    builder.add_cylinder_collision(
        radius=pin_radius, half_length=pin_half_length, density=stem_density, material=bearing
    )
    builder.add_cylinder_visual(radius=pin_radius, half_length=pin_half_length, material=_material(_PIN_COLOR))

    for sx in (-1.0, 1.0):
        collar_pose = sapien.Pose(p=[sx * collar_x, 0.0, 0.0])
        builder.add_cylinder_collision(
            pose=collar_pose,
            radius=collar_radius,
            half_length=collar_half_length,
            density=stem_density,
            material=bearing,
        )
        builder.add_cylinder_visual(
            pose=collar_pose,
            radius=collar_radius,
            half_length=collar_half_length,
            material=_material(_PIN_COLOR),
        )

    stem_pose = sapien.Pose(p=[0.0, 0.0, -stem_half_length])
    builder.add_box_collision(
        pose=stem_pose,
        half_size=[stem_half_x, stem_half_y, stem_half_length],
        density=stem_density,
        material=bearing,
    )
    builder.add_box_visual(
        pose=stem_pose,
        half_size=[stem_half_x, stem_half_y, stem_half_length],
        material=_material(_PENDANT_COLOR),
    )

    bob_pose = sapien.Pose(p=[0.0, 0.0, bob_center_z])
    builder.add_box_collision(pose=bob_pose, half_size=list(bob_half), density=bob_density, material=bearing)
    builder.add_box_visual(pose=bob_pose, half_size=list(bob_half), material=_material(_PENDANT_COLOR))

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
