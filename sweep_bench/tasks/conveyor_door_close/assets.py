from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_BODY_COLOR = (0.70, 0.71, 0.74, 1.0)
_INTERIOR_COLOR = (0.24, 0.28, 0.34, 1.0)
_JOURNAL_COLOR = (0.34, 0.36, 0.40, 1.0)
_LEAF_COLOR = (0.90, 0.44, 0.16, 1.0)
_RAIL_COLOR = (0.16, 0.34, 0.55, 1.0)
_PIN_COLOR = (0.55, 0.57, 0.60, 1.0)

_UPRIGHT_CYLINDER_Q = (math.cos(math.pi / 4.0), 0.0, math.sin(math.pi / 4.0), 0.0)


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


def build_cabinet_body(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    floor_half_thickness: float,
    wall_half_thickness: float,
    wall_top_z: float,
    roof_half_thickness: float,
    pin_x: float,
    pin_y: float,
    pin_radius: float,
    pin_clearance: float,
    journal_wall_half: float,
    journal_gap: float,
    stub_length: float,
    cap_half_thickness: float,
    strike_depth: float,
    strike_half_z: float,
    strike_hinge_gap: float,
    plate_half_thickness: float,
    plate_z_lo: float,
    plate_z_hi: float,
    density: float,
    bearing_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    clear = pin_radius + pin_clearance
    if abs(pin_y - (half_y + plate_half_thickness)) > 1e-9:
        raise ValueError(
            f"pin_y ({pin_y}) must equal half_y + plate_half_thickness "
            f"({half_y + plate_half_thickness}) so the shut leaf seats flat on the frame and clears it when swinging"
        )
    if plate_half_thickness >= clear:
        raise ValueError(
            f"plate_half_thickness ({plate_half_thickness}) must stay under the pocket clearance ({clear}): "
            "the leaf's hinge-end corner sweeps to pin_x + plate_half_thickness and would strike the stile"
        )
    if pin_x + clear + 2.0 * journal_wall_half > half_x:
        raise ValueError(f"the hinge stile at pin_x ({pin_x}) would overhang the footprint half-width ({half_x})")
    if plate_z_hi < wall_top_z + 2.0 * roof_half_thickness:
        raise ValueError(
            f"the leaf's top ({plate_z_hi}) must reach the roof top "
            f"({wall_top_z + 2.0 * roof_half_thickness}) or the doorway is not covered when shut"
        )

    builder = scene.create_actor_builder()
    bearing = bearing_material(bearing_friction)
    floor_top = 2.0 * floor_half_thickness
    wall_half_z = 0.5 * (wall_top_z - floor_top)
    wall_mid_z = floor_top + wall_half_z
    inner_half_x = half_x - 2.0 * wall_half_thickness

    _add_box(builder, (0.0, 0.0, floor_half_thickness), (half_x, half_y, floor_half_thickness), _BODY_COLOR, density)

    for sx in (-1.0, 1.0):
        _add_box(
            builder,
            (sx * (half_x - wall_half_thickness), 0.0, wall_mid_z),
            (wall_half_thickness, half_y, wall_half_z),
            _BODY_COLOR,
            density,
        )
    _add_box(
        builder,
        (0.0, -(half_y - wall_half_thickness), wall_mid_z),
        (inner_half_x, wall_half_thickness, wall_half_z),
        _INTERIOR_COLOR,
        density,
    )

    _add_box(
        builder,
        (0.0, 0.0, wall_top_z + roof_half_thickness),
        (half_x, half_y, roof_half_thickness),
        _BODY_COLOR,
        density,
    )

    strike_x_lo, strike_x_hi = -inner_half_x, pin_x - strike_hinge_gap
    _add_box(
        builder,
        (0.5 * (strike_x_lo + strike_x_hi), half_y - 0.5 * strike_depth, 0.5 * (floor_top + wall_top_z)),
        (0.5 * (strike_x_hi - strike_x_lo), 0.5 * strike_depth, strike_half_z),
        _BODY_COLOR,
        density,
    )

    gap = clear + journal_wall_half
    pocket_half = gap + journal_wall_half
    lo_band = (plate_z_lo - journal_gap - stub_length, plate_z_lo - journal_gap)
    hi_band = (plate_z_hi + journal_gap, plate_z_hi + journal_gap + stub_length)
    stile_lo, stile_hi = floor_top, hi_band[1] + 2.0 * cap_half_thickness
    _add_box(
        builder,
        (pin_x + gap, pin_y, 0.5 * (stile_lo + stile_hi)),
        (journal_wall_half, pocket_half, 0.5 * (stile_hi - stile_lo)),
        _JOURNAL_COLOR,
        density,
        bearing,
    )

    for z_lo, z_hi in (lo_band, hi_band):
        band_half_z = 0.5 * (z_hi - z_lo)
        band_mid_z = 0.5 * (z_hi + z_lo)
        _add_box(
            builder,
            (pin_x - gap, pin_y, band_mid_z),
            (journal_wall_half, pocket_half, band_half_z),
            _JOURNAL_COLOR,
            density,
            bearing,
        )
        for sy in (-1.0, 1.0):
            _add_box(
                builder,
                (pin_x, pin_y + sy * gap, band_mid_z),
                (pocket_half, journal_wall_half, band_half_z),
                _JOURNAL_COLOR,
                density,
                bearing,
            )
    _add_box(
        builder,
        (pin_x, pin_y, lo_band[0] - cap_half_thickness),
        (pocket_half, pocket_half, cap_half_thickness),
        _JOURNAL_COLOR,
        density,
        bearing,
    )
    _add_box(
        builder,
        (pin_x, pin_y, hi_band[1] + cap_half_thickness),
        (pocket_half, pocket_half, cap_half_thickness),
        _JOURNAL_COLOR,
        density,
        bearing,
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_cabinet_door(
    scene: ManiSkillScene,
    *,
    plate_half_length: float,
    plate_half_thickness: float,
    leaf_half_height: float,
    leaf_center_z: float,
    rail_r_lo: float,
    rail_half_height: float,
    rail_center_z: float,
    pin_radius: float,
    pin_half_length: float,
    density: float,
    bearing_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if pin_half_length <= leaf_half_height:
        raise ValueError(
            f"pin_half_length ({pin_half_length}) must exceed the leaf's half height ({leaf_half_height}): "
            "the journal pockets can only wrap a stub the leaf does not occupy"
        )
    if not 0.0 < rail_r_lo < 2.0 * plate_half_length:
        raise ValueError(f"rail_r_lo ({rail_r_lo}) must start inside the leaf's length ({2.0 * plate_half_length})")

    builder = scene.create_actor_builder()
    bearing = bearing_material(bearing_friction)

    pin_pose = sapien.Pose(q=list(_UPRIGHT_CYLINDER_Q))
    builder.add_cylinder_collision(
        pose=pin_pose, radius=pin_radius, half_length=pin_half_length, density=density, material=bearing
    )
    builder.add_cylinder_visual(
        pose=pin_pose, radius=pin_radius, half_length=pin_half_length, material=_material(_PIN_COLOR)
    )

    rail_half_length = 0.5 * (2.0 * plate_half_length - rail_r_lo)
    for half_x, center_x, half_z, center_z, color in (
        (plate_half_length, -plate_half_length, leaf_half_height, leaf_center_z, _LEAF_COLOR),
        (rail_half_length, -(rail_r_lo + rail_half_length), rail_half_height, rail_center_z, _RAIL_COLOR),
    ):
        pose = sapien.Pose(p=[center_x, 0.0, center_z])
        half = [half_x, plate_half_thickness, half_z]
        builder.add_box_collision(pose=pose, half_size=half, density=density, material=bearing)
        builder.add_box_visual(pose=pose, half_size=half, material=_material(color))

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
