from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_UPRIGHT = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def low_friction_material(friction: float) -> sapien.physx.PhysxMaterial:
    return sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)


def _add_box(
    builder: ActorBuilder,
    center: tuple[float, float, float],
    half: tuple[float, float, float],
    color: tuple[float, float, float, float],
    density: float,
    material: sapien.physx.PhysxMaterial | None = None,
    pose_q: tuple[float, float, float, float] | None = None,
) -> None:
    pose = sapien.Pose(p=list(center), q=list(pose_q) if pose_q is not None else [1.0, 0.0, 0.0, 0.0])
    builder.add_box_collision(pose=pose, half_size=half, density=density, material=material)
    builder.add_box_visual(pose=pose, half_size=half, material=_material(color))


def _plate_boxes(
    *,
    half_x: float,
    y_lo: float,
    y_hi: float,
    keyhole_y: float,
    open_x: float,
    open_half: float,
    channel_half_w: float,
    channel_x_hi: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    o_lo, o_hi = open_x - open_half, open_x + open_half
    ky_lo, ky_hi = keyhole_y - open_half, keyhole_y + open_half
    ch_lo, ch_hi = keyhole_y - channel_half_w, keyhole_y + channel_half_w
    return [
        ((-half_x, half_x), (y_lo, ky_lo)),
        ((-half_x, half_x), (ky_hi, y_hi)),
        ((-half_x, o_lo), (ky_lo, ky_hi)),
        ((channel_x_hi, half_x), (ky_lo, ky_hi)),
        ((o_hi, channel_x_hi), (ch_hi, ky_hi)),
        ((o_hi, channel_x_hi), (ky_lo, ch_lo)),
    ]


def _add_chamfer_lips(
    builder: ActorBuilder,
    *,
    keyhole_y: float,
    open_x: float,
    open_half: float,
    chamfer: float,
    plate_bottom: float,
    color: tuple[float, float, float, float],
    density: float,
    material: sapien.physx.PhysxMaterial,
) -> None:
    half_len = chamfer * math.sqrt(0.5)
    thick = 0.0015
    mid_r = open_half + 0.5 * chamfer
    mid_z = plate_bottom - 0.5 * chamfer
    off = thick * math.sqrt(0.5)

    q_y = (math.cos(math.radians(-67.5)), 0.0, math.sin(math.radians(-67.5)), 0.0)
    _add_box(
        builder,
        (open_x - mid_r - off, keyhole_y, mid_z + off),
        (thick, open_half, half_len),
        color,
        density,
        material,
        pose_q=q_y,
    )
    for sy, deg in ((1.0, -67.5), (-1.0, 67.5)):
        q_x = (math.cos(math.radians(deg)), math.sin(math.radians(deg)), 0.0, 0.0)
        _add_box(
            builder,
            (open_x, keyhole_y + sy * (mid_r + off), mid_z + off),
            (open_half, thick, half_len),
            color,
            density,
            material,
            pose_q=q_x,
        )


def build_keyed_carrier(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    base_half_thickness: float,
    cavity_height: float,
    plate_thickness: float,
    rail_thickness: float,
    keyhole_ys: Sequence[float],
    open_x: float,
    open_half: float,
    channel_half_w: float,
    channel_x_hi: float,
    chamfer: float,
    density: float,
    body_color: tuple[float, float, float, float],
    plate_color: tuple[float, float, float, float],
    plate_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if len(keyhole_ys) < 1:
        raise ValueError("build_keyed_carrier needs at least one keyhole y")
    if list(keyhole_ys) != sorted(keyhole_ys):
        raise ValueError(f"keyhole_ys must be sorted ascending, got {tuple(keyhole_ys)}")

    base_top = 2.0 * base_half_thickness
    plate_bottom = base_top + cavity_height
    mat = low_friction_material(plate_friction)

    builder = scene.create_actor_builder()

    _add_box(builder, (0.0, 0.0, base_half_thickness), (half_x, half_y, base_half_thickness), body_color, density)
    rail_half_z = 0.5 * (plate_bottom - base_top)
    for sx in (1.0, -1.0):
        _add_box(
            builder,
            (sx * (half_x - 0.5 * rail_thickness), 0.0, base_top + rail_half_z),
            (0.5 * rail_thickness, half_y, rail_half_z),
            body_color,
            density,
        )

    edges = [-half_y]
    edges += [0.5 * (a + b) for a, b in zip(keyhole_ys[:-1], keyhole_ys[1:], strict=True)]
    edges += [half_y]
    plate_half_z = 0.5 * plate_thickness
    plate_mid_z = plate_bottom + plate_half_z
    for i, keyhole_y in enumerate(keyhole_ys):
        for (x_lo, x_hi), (y_lo, y_hi) in _plate_boxes(
            half_x=half_x,
            y_lo=edges[i],
            y_hi=edges[i + 1],
            keyhole_y=keyhole_y,
            open_x=open_x,
            open_half=open_half,
            channel_half_w=channel_half_w,
            channel_x_hi=channel_x_hi,
        ):
            if x_hi - x_lo <= 0.0 or y_hi - y_lo <= 0.0:
                raise ValueError(f"degenerate plate box x=({x_lo}, {x_hi}) y=({y_lo}, {y_hi}) -- keyhole off the plate")
            _add_box(
                builder,
                (0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi), plate_mid_z),
                (0.5 * (x_hi - x_lo), 0.5 * (y_hi - y_lo), plate_half_z),
                plate_color,
                density,
                mat,
            )
        _add_chamfer_lips(
            builder,
            keyhole_y=keyhole_y,
            open_x=open_x,
            open_half=open_half,
            chamfer=chamfer,
            plate_bottom=plate_bottom,
            color=plate_color,
            density=density,
            material=mat,
        )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_studded_part(
    scene: ManiSkillScene,
    *,
    body_half_x: float,
    body_half_y: float,
    body_half_z: float,
    stud_radius: float,
    stud_length: float,
    head_half: float,
    head_half_thickness: float,
    color: tuple[float, float, float, float],
    density: float,
    body_friction: float,
    key_friction: float,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    body_mat = low_friction_material(body_friction)
    key_mat = low_friction_material(key_friction)
    mat = _material(color)

    body_pose = sapien.Pose()
    body_half = (body_half_x, body_half_y, body_half_z)
    builder.add_box_collision(pose=body_pose, half_size=body_half, density=density, material=body_mat)
    builder.add_box_visual(pose=body_pose, half_size=body_half, material=mat)

    stud_z = -(body_half_z + 0.5 * stud_length)
    stud_pose = sapien.Pose(p=[0.0, 0.0, stud_z], q=list(_UPRIGHT.q))
    builder.add_cylinder_collision(
        pose=stud_pose, radius=stud_radius, half_length=0.5 * stud_length, density=density, material=key_mat
    )
    builder.add_cylinder_visual(pose=stud_pose, radius=stud_radius, half_length=0.5 * stud_length, material=mat)

    head_z = -(body_half_z + stud_length + head_half_thickness)
    head_pose = sapien.Pose(p=[0.0, 0.0, head_z])
    head_size = (head_half, head_half, head_half_thickness)
    builder.add_box_collision(pose=head_pose, half_size=head_size, density=density, material=key_mat)
    builder.add_box_visual(pose=head_pose, half_size=head_size, material=mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
