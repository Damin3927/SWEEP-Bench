from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_JAR_COLOR = (0.35, 0.55, 0.45, 1.0)
_NECK_COLOR = (0.72, 0.78, 0.74, 1.0)
_BEAD_COLOR = (0.90, 0.30, 0.20, 1.0)
_CAP_COLOR = (0.20, 0.35, 0.75, 1.0)
_KNOB_COLOR = (0.95, 0.75, 0.15, 1.0)

_UPRIGHT = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])

_RING_OVERLAP = 1.25


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def build_bead_jar(
    scene: ManiSkillScene,
    *,
    body_radius: float,
    body_half_height: float,
    neck_radius: float,
    neck_height: float,
    bead_radius: float,
    bead_half_height: float,
    bead_below_neck_top: float,
    density: float = 1000.0,
    body_color: tuple[float, float, float, float] = _JAR_COLOR,
    neck_color: tuple[float, float, float, float] = _NECK_COLOR,
    bead_color: tuple[float, float, float, float] = _BEAD_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not neck_radius < bead_radius:
        raise ValueError(f"bead_radius={bead_radius} must exceed neck_radius={neck_radius}")
    if not bead_below_neck_top < neck_height:
        raise ValueError(f"bead_below_neck_top={bead_below_neck_top} must sit within the {neck_height} m neck")

    builder = scene.create_actor_builder()

    builder.add_cylinder_collision(pose=_UPRIGHT, radius=body_radius, half_length=body_half_height, density=density)
    builder.add_cylinder_visual(
        pose=_UPRIGHT, radius=body_radius, half_length=body_half_height, material=_material(body_color)
    )

    neck_pose = sapien.Pose(p=[0.0, 0.0, body_half_height + neck_height / 2.0], q=_UPRIGHT.q)
    builder.add_cylinder_collision(pose=neck_pose, radius=neck_radius, half_length=neck_height / 2.0, density=density)
    builder.add_cylinder_visual(
        pose=neck_pose, radius=neck_radius, half_length=neck_height / 2.0, material=_material(neck_color)
    )

    bead_z = body_half_height + neck_height - bead_below_neck_top
    bead_pose = sapien.Pose(p=[0.0, 0.0, bead_z], q=_UPRIGHT.q)
    builder.add_cylinder_visual(
        pose=bead_pose, radius=bead_radius, half_length=bead_half_height, material=_material(bead_color)
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_press_cap(
    scene: ManiSkillScene,
    *,
    bore_radius: float,
    outer_radius: float,
    skirt_height: float,
    lead_in_height: float,
    mouth_radius: float,
    crown_half_thickness: float,
    knob_radius: float,
    knob_half_height: float,
    segments: int = 12,
    bevel_half_thickness: float = 0.002,
    density: float = 1000.0,
    cap_color: tuple[float, float, float, float] = _CAP_COLOR,
    knob_color: tuple[float, float, float, float] = _KNOB_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not bore_radius < outer_radius:
        raise ValueError(f"bore_radius={bore_radius} must be smaller than outer_radius={outer_radius}")
    if not bore_radius < mouth_radius <= outer_radius:
        raise ValueError(f"mouth_radius={mouth_radius} must lie in ({bore_radius}, {outer_radius}]")
    if segments < 6:
        raise ValueError(f"segments={segments} is too coarse for a round bore (min 6)")

    cap_mat = _material(cap_color)
    knob_mat = _material(knob_color)
    builder = scene.create_actor_builder()

    knob_half = [knob_radius, knob_radius, knob_half_height]
    builder.add_box_collision(half_size=knob_half, density=density)
    builder.add_box_visual(half_size=knob_half, material=knob_mat)

    crown_z = -(knob_half_height + crown_half_thickness)
    crown_pose = sapien.Pose(p=[0.0, 0.0, crown_z], q=_UPRIGHT.q)
    builder.add_cylinder_collision(
        pose=crown_pose, radius=outer_radius, half_length=crown_half_thickness, density=density
    )
    builder.add_cylinder_visual(
        pose=crown_pose, radius=outer_radius, half_length=crown_half_thickness, material=cap_mat
    )

    ceiling_z = -(knob_half_height + 2.0 * crown_half_thickness)
    throat_z = ceiling_z - skirt_height

    pitch = 2.0 * math.pi / segments

    wall_radial_half = (outer_radius - bore_radius) / 2.0
    wall_half = [
        wall_radial_half,
        outer_radius * math.tan(pitch / 2.0) * _RING_OVERLAP,
        skirt_height / 2.0,
    ]
    wall_pose_0 = sapien.Pose(p=[bore_radius + wall_radial_half, 0.0, ceiling_z - skirt_height / 2.0])

    dr = mouth_radius - bore_radius
    alpha = math.atan2(lead_in_height, dr)
    slope_half = math.hypot(dr, lead_in_height) / 2.0
    bevel_r = (bore_radius + mouth_radius) / 2.0 + bevel_half_thickness * math.sin(alpha)
    bevel_z = throat_z - lead_in_height / 2.0 + bevel_half_thickness * math.cos(alpha)
    bevel_half = [slope_half, mouth_radius * math.tan(pitch / 2.0) * _RING_OVERLAP, bevel_half_thickness]
    bevel_pose_0 = sapien.Pose(
        p=[bevel_r, 0.0, bevel_z],
        q=[math.cos(alpha / 2.0), 0.0, math.sin(alpha / 2.0), 0.0],
    )

    for k in range(segments):
        yaw = k * pitch
        spin = sapien.Pose(q=[math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])
        wall_pose = spin * wall_pose_0
        builder.add_box_collision(pose=wall_pose, half_size=wall_half, density=density)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half, material=cap_mat)
        bevel_pose = spin * bevel_pose_0
        builder.add_box_collision(pose=bevel_pose, half_size=bevel_half, density=density)
        builder.add_box_visual(pose=bevel_pose, half_size=bevel_half, material=cap_mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
