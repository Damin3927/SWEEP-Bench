from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render
import torch

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_PLATE_COLOR = (0.55, 0.58, 0.62, 1.0)
_CHAMFER_COLOR = (0.95, 0.55, 0.20, 1.0)
_BIT_TIP_COLOR = (0.20, 0.42, 0.85, 1.0)
_BIT_SHANK_COLOR = (0.93, 0.96, 0.97, 1.0)

_HEX_SIDES = 6
HEX_PITCH = 2.0 * math.pi / _HEX_SIDES

_RING_OVERLAP = 1.25
_INV_SQRT3 = 1.0 / math.sqrt(3.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def fold_hex(angle: torch.Tensor) -> torch.Tensor:
    return angle - HEX_PITCH * torch.round(angle / HEX_PITCH)


def build_hex_socket_plate(
    scene: ManiSkillScene,
    *,
    socket_across_flats: float,
    mouth_across_flats: float,
    chamfer: float,
    socket_depth: float,
    plate_half_extent: float,
    floor_thickness: float = 0.005,
    ramp_half_thickness: float = 0.002,
    plate_color: tuple[float, float, float, float] = _PLATE_COLOR,
    chamfer_color: tuple[float, float, float, float] = _CHAMFER_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not socket_across_flats < mouth_across_flats:
        raise ValueError(
            f"socket_across_flats={socket_across_flats} must be smaller than mouth_across_flats={mouth_across_flats}"
        )
    if not mouth_across_flats / 2.0 < plate_half_extent:
        raise ValueError(
            f"mouth half-width {mouth_across_flats / 2.0} must be smaller than plate_half_extent={plate_half_extent}"
        )
    if chamfer <= 0.0 or socket_depth <= 0.0:
        raise ValueError(f"chamfer={chamfer} and socket_depth={socket_depth} must both be positive")

    a_socket = socket_across_flats / 2.0
    a_mouth = mouth_across_flats / 2.0
    half_height = (socket_depth + floor_thickness) / 2.0
    plate_mat = _material(plate_color)
    chamfer_mat = _material(chamfer_color)
    builder = scene.create_actor_builder()

    floor_half = [plate_half_extent, plate_half_extent, floor_thickness / 2.0]
    floor_pose = sapien.Pose(p=[0.0, 0.0, -half_height + floor_thickness / 2.0])
    builder.add_box_collision(pose=floor_pose, half_size=floor_half)
    builder.add_box_visual(pose=floor_pose, half_size=floor_half, material=plate_mat)

    wall_radial_half = (plate_half_extent - a_socket) / 2.0
    wall_tangential_half = plate_half_extent * math.tan(HEX_PITCH / 2.0) * _RING_OVERLAP
    wall_half = [wall_radial_half, wall_tangential_half, socket_depth / 2.0]
    wall_pose_0 = sapien.Pose(p=[a_socket + wall_radial_half, 0.0, half_height - socket_depth / 2.0])

    dr = a_mouth - a_socket
    slope_half = math.hypot(dr, chamfer) / 2.0
    alpha = math.atan2(chamfer, dr)
    ramp_r = (a_socket + a_mouth) / 2.0 + ramp_half_thickness * math.sin(alpha)
    ramp_z = half_height + chamfer / 2.0 - ramp_half_thickness * math.cos(alpha)
    ramp_half = [slope_half, a_mouth * math.tan(HEX_PITCH / 2.0) * _RING_OVERLAP, ramp_half_thickness]
    ramp_pose_0 = sapien.Pose(
        p=[ramp_r, 0.0, ramp_z],
        q=[math.cos(alpha / 2.0), 0.0, -math.sin(alpha / 2.0), 0.0],
    )

    for k in range(_HEX_SIDES):
        yaw = k * HEX_PITCH
        spin = sapien.Pose(q=[math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])
        wall_pose = spin * wall_pose_0
        builder.add_box_collision(pose=wall_pose, half_size=wall_half)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half, material=plate_mat)
        ramp_pose = spin * ramp_pose_0
        builder.add_box_collision(pose=ramp_pose, half_size=ramp_half)
        builder.add_box_visual(pose=ramp_pose, half_size=ramp_half, material=chamfer_mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_hex_bit(
    scene: ManiSkillScene,
    *,
    across_flats: float,
    half_length: float,
    tip_color: tuple[float, float, float, float] = _BIT_TIP_COLOR,
    shank_color: tuple[float, float, float, float] = _BIT_SHANK_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if across_flats <= 0.0 or half_length <= 0.0:
        raise ValueError(f"across_flats={across_flats} and half_length={half_length} must both be positive")

    a = across_flats / 2.0
    box_half = [a, a * _INV_SQRT3, half_length]
    tip_mat = _material(tip_color)
    shank_mat = _material(shank_color)
    half_visual = [a, a * _INV_SQRT3, half_length / 2.0]

    builder = scene.create_actor_builder()
    for k in range(3):
        yaw = HEX_PITCH / 2.0 + k * HEX_PITCH
        pose = sapien.Pose(q=[math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])
        builder.add_box_collision(pose=pose, half_size=box_half)
        tip_pose = sapien.Pose(p=[0.0, 0.0, half_length / 2.0], q=pose.q)
        shank_pose = sapien.Pose(p=[0.0, 0.0, -half_length / 2.0], q=pose.q)
        builder.add_box_visual(pose=tip_pose, half_size=half_visual, material=tip_mat)
        builder.add_box_visual(pose=shank_pose, half_size=half_visual, material=shank_mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
