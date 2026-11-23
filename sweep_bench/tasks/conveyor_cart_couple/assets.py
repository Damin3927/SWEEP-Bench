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

_LEAD_COLOR = (0.22, 0.42, 0.66, 1.0)
_FOLLOWER_COLOR = (0.86, 0.58, 0.14, 1.0)
_STUB_COLOR = (0.28, 0.30, 0.34, 1.0)
_COUPLING_COLOR = (0.55, 0.58, 0.62, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def _rot_x(angle: float) -> list[float]:
    return [math.cos(angle * 0.5), math.sin(angle * 0.5), 0.0, 0.0]


def _rot_z(angle: float) -> list[float]:
    return [math.cos(angle * 0.5), 0.0, 0.0, math.sin(angle * 0.5)]


def _add_box(
    builder: ActorBuilder,
    *,
    pose: sapien.Pose,
    half_size: tuple[float, float, float],
    color: tuple[float, float, float, float],
    density: float,
    material: PhysxMaterial | None = None,
) -> None:
    builder.add_box_collision(pose=pose, half_size=half_size, material=material, density=density)
    builder.add_box_visual(pose=pose, half_size=half_size, material=_material(color))


def build_lead_cart(
    scene: ManiSkillScene,
    *,
    body_half_x: float,
    body_half_y: float,
    body_height: float,
    socket_depth: float,
    channel_half_x: float,
    channel_floor_z: float,
    crest_z: float,
    crest_y: float,
    mouth_flare: float,
    mouth_flare_run: float,
    stub_half_x: float,
    stub_half_y: float,
    stub_half_z: float,
    density: float,
    coupling_friction: float,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    coupling = PhysxMaterial(static_friction=coupling_friction, dynamic_friction=coupling_friction, restitution=0.0)
    oz = body_height + stub_half_z

    mouth_y = -body_half_y - socket_depth

    _add_box(
        builder,
        pose=sapien.Pose(p=[0.0, 0.0, body_height * 0.5 - oz]),
        half_size=(body_half_x, body_half_y, body_height * 0.5),
        color=_LEAD_COLOR,
        density=density,
    )

    _add_box(
        builder,
        pose=sapien.Pose(p=[0.0, mouth_y * 0.5 - body_half_y * 0.5, channel_floor_z * 0.5 - oz]),
        half_size=(body_half_x, socket_depth * 0.5, channel_floor_z * 0.5),
        color=_LEAD_COLOR,
        density=density,
        material=coupling,
    )

    wall_half_x = (body_half_x - channel_half_x) * 0.5
    wall_half_z = (body_height - channel_floor_z) * 0.5
    wall_start_y = mouth_y + mouth_flare_run
    wall_half_y = (-body_half_y - wall_start_y) * 0.5
    for sx in (-1.0, 1.0):
        _add_box(
            builder,
            pose=sapien.Pose(
                p=[
                    sx * (channel_half_x + wall_half_x),
                    wall_start_y + wall_half_y,
                    channel_floor_z + wall_half_z - oz,
                ]
            ),
            half_size=(wall_half_x, wall_half_y, wall_half_z),
            color=_COUPLING_COLOR,
            density=density,
            material=coupling,
        )

    flare_angle = math.atan2(mouth_flare, mouth_flare_run)
    flare_len = math.hypot(mouth_flare, mouth_flare_run)
    flare_half_x = 0.004
    for sx in (-1.0, 1.0):
        mid_x = sx * (channel_half_x + mouth_flare * 0.5)
        mid_y = mouth_y + mouth_flare_run * 0.5
        _add_box(
            builder,
            pose=sapien.Pose(
                p=[
                    mid_x + sx * flare_half_x * math.cos(flare_angle),
                    mid_y + flare_half_x * math.sin(flare_angle),
                    channel_floor_z + wall_half_z - oz,
                ],
                q=_rot_z(sx * flare_angle),
            ),
            half_size=(flare_half_x, flare_len * 0.5, wall_half_z),
            color=_COUPLING_COLOR,
            density=density,
            material=coupling,
        )

    ramp_run = crest_y - mouth_y
    ramp_rise = crest_z - channel_floor_z
    ramp_angle = math.atan2(ramp_rise, ramp_run)
    ramp_half_t = 0.002
    _add_box(
        builder,
        pose=sapien.Pose(
            p=[
                0.0,
                (mouth_y + crest_y) * 0.5 + ramp_half_t * math.sin(ramp_angle),
                (channel_floor_z + crest_z) * 0.5 - ramp_half_t * math.cos(ramp_angle) - oz,
            ],
            q=_rot_x(ramp_angle),
        ),
        half_size=(channel_half_x, math.hypot(ramp_run, ramp_rise) * 0.5, ramp_half_t),
        color=_COUPLING_COLOR,
        density=density,
        material=coupling,
    )

    _add_box(
        builder,
        pose=sapien.Pose(p=[0.0, 0.0, 0.0]),
        half_size=(stub_half_x, stub_half_y, stub_half_z),
        color=_STUB_COLOR,
        density=density,
    )
    return builder


def build_follower_cart(
    scene: ManiSkillScene,
    *,
    body_half_x: float,
    body_half_y: float,
    body_height: float,
    blade_half_x: float,
    blade_reach: float,
    blade_bottom_z: float,
    blade_top_z: float,
    toe_len: float,
    toe_bottom_z: float,
    density: float,
    coupling_friction: float,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    coupling = PhysxMaterial(static_friction=coupling_friction, dynamic_friction=coupling_friction, restitution=0.0)
    oz = body_height * 0.5

    _add_box(
        builder,
        pose=sapien.Pose(p=[0.0, 0.0, 0.0]),
        half_size=(body_half_x, body_half_y, body_height * 0.5),
        color=_FOLLOWER_COLOR,
        density=density,
    )

    blade_half_y = blade_reach * 0.5
    _add_box(
        builder,
        pose=sapien.Pose(p=[0.0, body_half_y + blade_half_y, (blade_bottom_z + blade_top_z) * 0.5 - oz]),
        half_size=(blade_half_x, blade_half_y, (blade_top_z - blade_bottom_z) * 0.5),
        color=_COUPLING_COLOR,
        density=density,
        material=coupling,
    )

    tip_y = body_half_y + blade_reach
    _add_box(
        builder,
        pose=sapien.Pose(p=[0.0, tip_y - toe_len * 0.5, (toe_bottom_z + blade_bottom_z) * 0.5 - oz]),
        half_size=(blade_half_x, toe_len * 0.5, (blade_bottom_z - toe_bottom_z) * 0.5),
        color=_COUPLING_COLOR,
        density=density,
        material=coupling,
    )
    return builder
