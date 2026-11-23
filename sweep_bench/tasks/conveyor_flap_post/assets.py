from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_BODY_COLOR = (0.66, 0.68, 0.72, 1.0)
_JOURNAL_COLOR = (0.32, 0.34, 0.38, 1.0)
_DOOR_COLOR = (0.92, 0.45, 0.16, 1.0)
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


def build_flap_bin(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    wall_half_thickness: float,
    floor_half_thickness: float,
    wall_top_z: float,
    pin_y: float,
    pin_z: float,
    pin_radius: float,
    pin_clearance: float,
    journal_inner_x: float,
    journal_outer_x: float,
    journal_base_z: float,
    stop_inset: float,
    stop_top_z: float,
    plate_half_x: float,
    density: float,
    bearing_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    clear = pin_radius + pin_clearance
    if pin_y != half_y - 2.0 * wall_half_thickness:
        raise ValueError(
            f"pin_y ({pin_y}) must equal the wall inner face {half_y - 2.0 * wall_half_thickness} "
            "or a swinging door clips the wall"
        )
    if wall_top_z >= pin_z:
        raise ValueError(f"wall_top_z ({wall_top_z}) must sit below the hinge axis ({pin_z})")
    if not journal_inner_x < journal_outer_x <= half_x:
        raise ValueError(f"journal x span must increase and fit the footprint: [{journal_inner_x}, {journal_outer_x}]")
    if journal_base_z >= pin_z - clear:
        raise ValueError(f"journal_base_z ({journal_base_z}) must sit below the groove floor ({pin_z - clear})")
    if not 0.0 < stop_top_z < pin_z:
        raise ValueError(f"stop_top_z ({stop_top_z}) must sit between the floor and the hinge axis ({pin_z})")

    builder = scene.create_actor_builder()

    _add_box(builder, (0.0, 0.0, floor_half_thickness), (half_x, half_y, floor_half_thickness), _BODY_COLOR, density)

    floor_top = 2.0 * floor_half_thickness
    wall_half_z = 0.5 * (wall_top_z - floor_top)
    wall_mid_z = floor_top + wall_half_z

    for sx in (-1.0, 1.0):
        _add_box(
            builder,
            (sx * (half_x - wall_half_thickness), 0.0, wall_mid_z),
            (wall_half_thickness, half_y, wall_half_z),
            _BODY_COLOR,
            density,
        )
    for sy in (-1.0, 1.0):
        _add_box(
            builder,
            (0.0, sy * (half_y - wall_half_thickness), wall_mid_z),
            (half_x - 2.0 * wall_half_thickness, wall_half_thickness, wall_half_z),
            _BODY_COLOR,
            density,
        )

    stop_half_z = 0.5 * (stop_top_z - floor_top)
    for sy in (-1.0, 1.0):
        _add_box(
            builder,
            (0.0, sy * (pin_y - 0.5 * stop_inset), floor_top + stop_half_z),
            (plate_half_x, 0.5 * stop_inset, stop_half_z),
            _BODY_COLOR,
            density,
        )

    bearing = bearing_material(bearing_friction)
    journal_half_x = 0.5 * (journal_outer_x - journal_inner_x)
    journal_mid_x = 0.5 * (journal_outer_x + journal_inner_x)
    pocket_half_y = clear + 2.0 * wall_half_thickness
    pocket_wall_half = wall_half_thickness
    cap_half_z = 0.5 * wall_half_thickness
    for sy in (-1.0, 1.0):
        py = sy * pin_y
        for sx in (-1.0, 1.0):
            jx = sx * journal_mid_x
            ledge_half_z = 0.5 * (pin_z - clear - journal_base_z)
            _add_box(
                builder,
                (jx, py, journal_base_z + ledge_half_z),
                (journal_half_x, pocket_half_y, ledge_half_z),
                _JOURNAL_COLOR,
                density,
                bearing,
            )
            for sj in (-1.0, 1.0):
                _add_box(
                    builder,
                    (jx, py + sj * (clear + pocket_wall_half), pin_z),
                    (journal_half_x, pocket_wall_half, clear),
                    _JOURNAL_COLOR,
                    density,
                    bearing,
                )
            _add_box(
                builder,
                (jx, py, pin_z + clear + cap_half_z),
                (journal_half_x, pocket_half_y, cap_half_z),
                _JOURNAL_COLOR,
                density,
                bearing,
            )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_flap_door(
    scene: ManiSkillScene,
    *,
    plate_half_x: float,
    plate_half_y: float,
    plate_half_thickness: float,
    plate_root_gap: float,
    pin_radius: float,
    pin_half_length: float,
    density: float,
    bearing_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if plate_root_gap <= 0.0:
        raise ValueError(f"plate_root_gap ({plate_root_gap}) must be positive so the plate root clears the journals")

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
        material=_material(_DOOR_COLOR),
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
