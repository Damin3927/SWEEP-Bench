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

_RAMP_OVERLAP = 1.15


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def build_bead_jug(
    scene: ManiSkillScene,
    *,
    inner_half_x: float,
    inner_half_y: float,
    inner_height: float,
    wall_thickness: float,
    grip_wall_thickness: float,
    floor_thickness: float,
    lid_thickness: float,
    spout_lip: float,
    spout_height: float,
    weir_angle: float,
    weir_thickness: float,
    color: tuple[float, float, float, float],
    density: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not spout_lip > 0.0:
        raise ValueError(f"spout_lip={spout_lip} must be positive (a floor-level spout dumps at any tilt)")
    if not spout_lip + spout_height < inner_height:
        raise ValueError(
            f"spout_lip={spout_lip} + spout_height={spout_height} must stay under inner_height={inner_height}, "
            "or the spout is a rim notch rather than a spout"
        )

    outer_half_x = inner_half_x + wall_thickness
    outer_half_y = inner_half_y + grip_wall_thickness
    half_z = 0.5 * (floor_thickness + inner_height + lid_thickness)
    floor_top = -half_z + floor_thickness
    mat = _material(color)
    grip_mat = PhysxMaterial(static_friction=1.0, dynamic_friction=0.9, restitution=0.0)

    builder = scene.create_actor_builder()

    def add_box(half: list[float], p: list[float], material: PhysxMaterial | None = None) -> None:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=half, material=material, density=density)
        builder.add_box_visual(pose=pose, half_size=half, material=mat)

    add_box([outer_half_x, outer_half_y, 0.5 * floor_thickness], [0.0, 0.0, -half_z + 0.5 * floor_thickness])
    add_box([outer_half_x, outer_half_y, 0.5 * lid_thickness], [0.0, 0.0, half_z - 0.5 * lid_thickness])
    for sy in (1.0, -1.0):
        add_box(
            [inner_half_x, 0.5 * grip_wall_thickness, 0.5 * inner_height],
            [0.0, sy * (inner_half_y + 0.5 * grip_wall_thickness), floor_top + 0.5 * inner_height],
            grip_mat,
        )
    add_box(
        [0.5 * wall_thickness, inner_half_y, 0.5 * inner_height],
        [inner_half_x + 0.5 * wall_thickness, 0.0, floor_top + 0.5 * inner_height],
    )
    add_box(
        [0.5 * wall_thickness, inner_half_y, 0.5 * spout_lip],
        [-(inner_half_x + 0.5 * wall_thickness), 0.0, floor_top + 0.5 * spout_lip],
    )
    header = inner_height - spout_lip - spout_height
    add_box(
        [0.5 * wall_thickness, inner_half_y, 0.5 * header],
        [-(inner_half_x + 0.5 * wall_thickness), 0.0, floor_top + inner_height - 0.5 * header],
    )

    run = spout_lip / math.tan(weir_angle)
    plate_len = math.hypot(run, spout_lip) + weir_thickness
    normal_x, normal_z = math.sin(weir_angle), math.cos(weir_angle)
    mid_x = -inner_half_x + 0.5 * run
    mid_z = floor_top + 0.5 * spout_lip
    weir_pose = sapien.Pose(
        p=[
            mid_x - 0.5 * weir_thickness * normal_x,
            0.0,
            mid_z - 0.5 * weir_thickness * normal_z,
        ],
        q=[math.cos(0.5 * weir_angle), 0.0, math.sin(0.5 * weir_angle), 0.0],
    )
    weir_half = [0.5 * plate_len, inner_half_y, 0.5 * weir_thickness]
    builder.add_box_collision(pose=weir_pose, half_size=weir_half, density=density)
    builder.add_box_visual(pose=weir_pose, half_size=weir_half, material=mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_funnel_hopper(
    scene: ManiSkillScene,
    *,
    well_half: float,
    well_depth: float,
    wall_thickness: float,
    floor_thickness: float,
    cone_depth: float,
    mouth_half: float,
    body_color: tuple[float, float, float, float],
    funnel_color: tuple[float, float, float, float],
    density: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not well_half + wall_thickness < mouth_half:
        raise ValueError(
            f"mouth_half={mouth_half} must exceed the outer well half-extent "
            f"{well_half + wall_thickness} or the funnel slopes the wrong way"
        )

    outer_half = well_half + wall_thickness
    body_mat = _material(body_color)
    funnel_mat = _material(funnel_color)
    builder = scene.create_actor_builder()

    def add_box(
        half: list[float],
        pose: sapien.Pose,
        material: sapien.render.RenderMaterial,
    ) -> None:
        builder.add_box_collision(pose=pose, half_size=half, density=density)
        builder.add_box_visual(pose=pose, half_size=half, material=material)

    add_box(
        [outer_half, outer_half, 0.5 * floor_thickness],
        sapien.Pose(p=[0.0, 0.0, 0.5 * floor_thickness]),
        body_mat,
    )

    wall_z = floor_thickness + 0.5 * well_depth
    for sx in (1.0, -1.0):
        add_box(
            [0.5 * wall_thickness, outer_half, 0.5 * well_depth],
            sapien.Pose(p=[sx * (well_half + 0.5 * wall_thickness), 0.0, wall_z]),
            body_mat,
        )
    for sy in (1.0, -1.0):
        add_box(
            [well_half, 0.5 * wall_thickness, 0.5 * well_depth],
            sapien.Pose(p=[0.0, sy * (well_half + 0.5 * wall_thickness), wall_z]),
            body_mat,
        )

    ramp_thickness = 0.004
    dr = mouth_half - outer_half
    slope_half = 0.5 * math.hypot(dr, cone_depth)
    alpha = math.atan2(cone_depth, dr)
    ramp_r = 0.5 * (outer_half + mouth_half) + 0.5 * ramp_thickness * math.sin(alpha)
    ramp_z = floor_thickness + well_depth + 0.5 * cone_depth - 0.5 * ramp_thickness * math.cos(alpha)
    ramp_half = [slope_half, mouth_half * _RAMP_OVERLAP, 0.5 * ramp_thickness]
    ramp_pose_0 = sapien.Pose(
        p=[ramp_r, 0.0, ramp_z],
        q=[math.cos(0.5 * alpha), 0.0, -math.sin(0.5 * alpha), 0.0],
    )
    for k in range(4):
        yaw = k * 0.5 * math.pi
        spin = sapien.Pose(q=[math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])
        add_box(ramp_half, spin * ramp_pose_0, funnel_mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_bead(
    scene: ManiSkillScene,
    *,
    radius: float,
    color: tuple[float, float, float, float],
    density: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    material = PhysxMaterial(static_friction=0.3, dynamic_friction=0.3, restitution=0.0)
    builder.add_sphere_collision(radius=radius, material=material, density=density)
    builder.add_sphere_visual(radius=radius, material=_material(color))
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
