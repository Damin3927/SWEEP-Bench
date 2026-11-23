from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_BLOCK_COLOR = (0.62, 0.66, 0.70, 1.0)
_BORE_COLOR = (0.15, 0.62, 0.62, 1.0)
_SHROUD_COLOR = (0.95, 0.55, 0.20, 1.0)

_RING_OVERLAP = 1.25


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def _port_pose(tilt_rad: float, offset_x: float) -> sapien.Pose:
    half = -tilt_rad / 2.0
    return sapien.Pose(p=[offset_x, 0.0, 0.0], q=[math.cos(half), 0.0, math.sin(half), 0.0])


def validate_port_geometry(
    *,
    bore_radius: float,
    bore_length: float,
    lead_depth: float,
    mouth_radius: float,
    tilt_rad: float,
    wall_thickness: float,
    tube_foot: float,
    base_thickness: float,
    port_offset_x: float,
    plate_half_extent: float,
    shroud_gap: float,
    shroud_overhang: float,
    shroud_half_y: float,
) -> None:
    if not bore_radius < mouth_radius:
        raise ValueError(f"bore_radius={bore_radius} must be smaller than mouth_radius={mouth_radius}")
    if not math.radians(20.0) <= tilt_rad <= math.radians(60.0):
        raise ValueError(f"tilt_rad={tilt_rad} is outside the steep-oblique band this fixture is built for")

    sin_t, cos_t, tan_t = math.sin(tilt_rad), math.cos(tilt_rad), math.tan(tilt_rad)

    mouth_x_half = mouth_radius * cos_t
    if shroud_overhang < mouth_x_half:
        raise ValueError(
            f"shroud_overhang={shroud_overhang:.4f} does not cover the mouth: the mouth disc reaches "
            f"{mouth_x_half:.4f} m to -x of its centre, so a vertical descent would slip past the edge"
        )
    corridor_half = bore_radius / cos_t
    if shroud_gap * tan_t <= shroud_overhang + corridor_half:
        raise ValueError(
            f"shroud_gap={shroud_gap:.4f} is too small: at the shroud's underside the along-axis corridor "
            f"spans up to {shroud_gap * tan_t - corridor_half:.4f} m to -x of the mouth, but the shroud "
            f"already reaches {shroud_overhang:.4f} m -- the shroud would block the approach it must allow"
        )
    if shroud_half_y < mouth_radius:
        raise ValueError(f"shroud_half_y={shroud_half_y} must cover the mouth's {mouth_radius} m half-width")

    _check_tube_foot(
        tube_foot=tube_foot,
        outer_radius=bore_radius + wall_thickness,
        base_thickness=base_thickness,
        sin_t=sin_t,
        cos_t=cos_t,
    )

    mouth_x = port_offset_x - (bore_length + lead_depth) * sin_t
    shroud_edge_x = mouth_x - shroud_overhang
    if shroud_edge_x < -plate_half_extent:
        raise ValueError(
            f"the shroud's edge reaches x={shroud_edge_x:.4f} but the plate ends at {-plate_half_extent:.4f}: "
            "raise port_offset_x or widen the plate"
        )
    post_x_inner = port_offset_x + (bore_radius + wall_thickness) / cos_t
    if post_x_inner >= plate_half_extent:
        raise ValueError(
            f"no room for the shroud post: the tube reaches x={post_x_inner:.4f} and the plate ends at "
            f"{plate_half_extent:.4f}"
        )


def build_shrouded_port_block(
    scene: ManiSkillScene,
    *,
    bore_radius: float,
    bore_length: float,
    lead_depth: float,
    mouth_radius: float,
    tilt_rad: float,
    wall_thickness: float,
    tube_foot: float,
    plate_half_extent: float,
    base_thickness: float,
    port_offset_x: float,
    shroud_gap: float,
    shroud_overhang: float,
    shroud_half_y: float,
    shroud_half_thickness: float,
    post_half_y: float,
    density: float,
    segments: int = 12,
    ramp_half_thickness: float = 0.002,
    block_color: tuple[float, float, float, float] = _BLOCK_COLOR,
    bore_color: tuple[float, float, float, float] = _BORE_COLOR,
    shroud_color: tuple[float, float, float, float] = _SHROUD_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if segments < 6:
        raise ValueError(f"segments={segments} is too coarse for a round bore (min 6)")
    validate_port_geometry(
        bore_radius=bore_radius,
        bore_length=bore_length,
        lead_depth=lead_depth,
        mouth_radius=mouth_radius,
        tilt_rad=tilt_rad,
        wall_thickness=wall_thickness,
        tube_foot=tube_foot,
        base_thickness=base_thickness,
        port_offset_x=port_offset_x,
        plate_half_extent=plate_half_extent,
        shroud_gap=shroud_gap,
        shroud_overhang=shroud_overhang,
        shroud_half_y=shroud_half_y,
    )

    sin_t, cos_t = math.sin(tilt_rad), math.cos(tilt_rad)
    mouth_axial = bore_length + lead_depth
    mouth_x = port_offset_x - mouth_axial * sin_t
    mouth_z = mouth_axial * cos_t

    shroud_edge_x = mouth_x - shroud_overhang
    post_x_inner = port_offset_x + (bore_radius + wall_thickness) / cos_t

    block_mat = _material(block_color)
    bore_mat = _material(bore_color)
    shroud_mat = _material(shroud_color)
    builder = scene.create_actor_builder()

    def slab(center: tuple[float, float, float], half: tuple[float, float, float], mat) -> None:
        pose = sapien.Pose(p=list(center))
        builder.add_box_collision(pose=pose, half_size=list(half), density=density)
        builder.add_box_visual(pose=pose, half_size=list(half), material=mat)

    slab(
        (0.0, 0.0, -base_thickness / 2.0),
        (plate_half_extent, plate_half_extent, base_thickness / 2.0),
        block_mat,
    )

    shroud_under_z = mouth_z + shroud_gap
    slab(
        ((post_x_inner + plate_half_extent) / 2.0, 0.0, shroud_under_z / 2.0),
        ((plate_half_extent - post_x_inner) / 2.0, post_half_y, shroud_under_z / 2.0),
        block_mat,
    )
    slab(
        ((shroud_edge_x + plate_half_extent) / 2.0, 0.0, shroud_under_z + shroud_half_thickness),
        ((plate_half_extent - shroud_edge_x) / 2.0, shroud_half_y, shroud_half_thickness),
        shroud_mat,
    )

    port = _port_pose(tilt_rad, port_offset_x)
    pitch = 2.0 * math.pi / segments

    tube_top = bore_length
    tube_bottom = -tube_foot
    wall_half = [
        wall_thickness / 2.0,
        (bore_radius + wall_thickness) * math.tan(pitch / 2.0) * _RING_OVERLAP,
        (tube_top - tube_bottom) / 2.0,
    ]
    wall_pose_0 = sapien.Pose(p=[bore_radius + wall_thickness / 2.0, 0.0, (tube_top + tube_bottom) / 2.0])

    dr = mouth_radius - bore_radius
    slope_half = math.hypot(dr, lead_depth) / 2.0
    alpha = math.atan2(lead_depth, dr)
    ramp_r = (bore_radius + mouth_radius) / 2.0 + ramp_half_thickness * math.sin(alpha)
    ramp_z = bore_length + lead_depth / 2.0 - ramp_half_thickness * math.cos(alpha)
    ramp_half = [slope_half, mouth_radius * math.tan(pitch / 2.0) * _RING_OVERLAP, ramp_half_thickness]
    ramp_pose_0 = sapien.Pose(
        p=[ramp_r, 0.0, ramp_z],
        q=[math.cos(alpha / 2.0), 0.0, -math.sin(alpha / 2.0), 0.0],
    )

    for k in range(segments):
        yaw = k * pitch
        spin = sapien.Pose(q=[math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])
        wall_pose = port * spin * wall_pose_0
        builder.add_box_collision(pose=wall_pose, half_size=wall_half, density=density)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half, material=block_mat)
        ramp_pose = port * spin * ramp_pose_0
        builder.add_box_collision(pose=ramp_pose, half_size=ramp_half, density=density)
        builder.add_box_visual(pose=ramp_pose, half_size=ramp_half, material=bore_mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def _check_tube_foot(
    *,
    tube_foot: float,
    outer_radius: float,
    base_thickness: float,
    sin_t: float,
    cos_t: float,
) -> None:
    axis_z = -tube_foot * cos_t
    high = axis_z + outer_radius * sin_t
    low = axis_z - outer_radius * sin_t
    if high > 0.0:
        raise ValueError(
            f"tube_foot={tube_foot:.4f} leaves the stub's heel {high:.4f} m above the plate's top face: "
            "the bore would vent through the gap under it"
        )
    if low < -base_thickness:
        raise ValueError(
            f"tube_foot={tube_foot:.4f} drives the stub {-(low + base_thickness):.4f} m through the plate's "
            "underside: the block would ride on the pipe, not on its foot"
        )
