from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render
from sapien.physx import PhysxMaterial

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_PAD_HALF_HEIGHT = 0.00925
_PAD_HALF_ALONG = 0.0076

_MIN_PAD_WALL_CLEARANCE = 0.004
_MIN_POCKET_PLAY = 0.010

_BASE_COLOR = (0.40, 0.43, 0.48, 1.0)
_RAMP_COLOR = (0.72, 0.74, 0.78, 1.0)
_POCKET_COLOR = (0.30, 0.52, 0.42, 1.0)

CARRIER_DENSITY = 1800.0
_FRICTION = 0.3


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6, metallic=0.1)


def ramp_surface_z(local_y: float, angle: float) -> float:
    return -local_y * math.tan(angle)


def seat_height(radius: float, angle: float) -> float:
    return radius / math.cos(angle)


def pad_wall_clearance(radius: float, angle: float, release_gap: float, wall_height: float) -> float:
    return seat_height(radius, angle) + release_gap - _PAD_HALF_HEIGHT - wall_height


def build_ramp_carrier(
    scene: ManiSkillScene,
    *,
    ball_radius: float,
    angle: float,
    run_above: float,
    run_below: float,
    pocket_length: float,
    pocket_depth: float,
    pocket_rim_above_ramp: float,
    channel_side_gap: float,
    channel_wall_height: float,
    wall_thickness: float,
    base_thickness: float,
    ramp_thickness: float,
    release_gap: float,
    density: float = CARRIER_DENSITY,
    friction: float = _FRICTION,
) -> ActorBuilder:
    clearance = pad_wall_clearance(ball_radius, angle, release_gap, channel_wall_height)
    if clearance < _MIN_PAD_WALL_CLEARANCE:
        raise ValueError(
            f"a {ball_radius} m ball leaves only {clearance:.4f} m between the fingertip pads at the release "
            f"height and the {channel_wall_height} m channel walls (need {_MIN_PAD_WALL_CLEARANCE} m); the jaws "
            f"close ACROSS the channel, so they must pass OVER the walls"
        )
    if pocket_depth < ball_radius:
        raise ValueError(
            f"catch well is {pocket_depth} m deep against a {ball_radius} m ball radius: the well must be at "
            f"least one radius deep, or the catch degenerates into a floor-plus-lip that cannot retain the ball"
        )
    if pocket_length < 2.0 * ball_radius + _MIN_POCKET_PLAY:
        raise ValueError(
            f"catch well is {pocket_length} m long against a {2 * ball_radius} m ball (need "
            f"{_MIN_POCKET_PLAY} m of play); the ball would wedge on the rim instead of landing on the floor"
        )

    half_x = ball_radius + channel_side_gap
    ramp_bottom_z = ramp_surface_z(run_below, angle)
    floor_z = ramp_bottom_z - pocket_depth
    pocket_y_lo = run_below
    pocket_y_hi = run_below + pocket_length
    base_y_lo = -run_above - 2.0 * wall_thickness
    base_y_hi = pocket_y_hi + wall_thickness
    base_half_x = half_x + wall_thickness + 0.003

    mat_base = _material(_BASE_COLOR)
    mat_ramp = _material(_RAMP_COLOR)
    mat_pocket = _material(_POCKET_COLOR)
    phys = PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)

    builder = scene.create_actor_builder()

    def _box(pose: sapien.Pose, half: list[float], mat: sapien.render.RenderMaterial) -> None:
        builder.add_box_collision(pose=pose, half_size=half, material=phys, density=density)
        builder.add_box_visual(pose=pose, half_size=half, material=mat)

    _box(
        sapien.Pose(p=[0.0, 0.5 * (base_y_lo + base_y_hi), floor_z - 0.5 * base_thickness]),
        [base_half_x, 0.5 * (base_y_hi - base_y_lo), 0.5 * base_thickness],
        mat_base,
    )

    sa, ca = math.sin(angle), math.cos(angle)
    slope_half = 0.5 * (run_above + run_below) / ca
    mid_y = 0.5 * (run_below - run_above)
    mid_z = ramp_surface_z(mid_y, angle)
    q_ramp = [math.cos(0.5 * angle), -math.sin(0.5 * angle), 0.0, 0.0]
    _box(
        sapien.Pose(p=[0.0, mid_y - 0.5 * ramp_thickness * sa, mid_z - 0.5 * ramp_thickness * ca], q=q_ramp),
        [half_x, slope_half, 0.5 * ramp_thickness],
        mat_ramp,
    )

    for sign in (1.0, -1.0):
        _box(
            sapien.Pose(
                p=[
                    sign * (half_x + 0.5 * wall_thickness),
                    mid_y + 0.5 * channel_wall_height * sa,
                    mid_z + 0.5 * channel_wall_height * ca,
                ],
                q=q_ramp,
            ),
            [0.5 * wall_thickness, slope_half, 0.5 * channel_wall_height],
            mat_ramp,
        )

    pocket_wall_top = ramp_bottom_z + pocket_rim_above_ramp
    wall_half_z = 0.5 * (pocket_wall_top - floor_z)
    wall_mid_z = floor_z + wall_half_z
    _box(
        sapien.Pose(p=[0.0, pocket_y_hi + 0.5 * wall_thickness, wall_mid_z]),
        [base_half_x, 0.5 * wall_thickness, wall_half_z],
        mat_pocket,
    )
    for sign in (1.0, -1.0):
        _box(
            sapien.Pose(p=[sign * (half_x + 0.5 * wall_thickness), 0.5 * (pocket_y_lo + pocket_y_hi), wall_mid_z]),
            [0.5 * wall_thickness, 0.5 * pocket_length, wall_half_z],
            mat_pocket,
        )

    return builder


def build_ball(
    scene: ManiSkillScene,
    *,
    radius: float,
    color: tuple[float, float, float, float],
    density: float = 1000.0,
    friction: float = _FRICTION,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    material = PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    builder.add_sphere_collision(radius=radius, material=material, density=density)
    builder.add_sphere_visual(radius=radius, material=_material(color))
    return builder
