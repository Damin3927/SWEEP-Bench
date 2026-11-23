from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_BASE_COLOR = (0.45, 0.47, 0.50, 1.0)
_PIN_COLOR = (0.90, 0.55, 0.15, 1.0)
_BRACKET_COLOR = (0.30, 0.45, 0.72, 1.0)

_UPRIGHT = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])

_CONE_OVERLAP = 1.25
_CONE_PLATE_HALF_THICKNESS = 0.0012
_RAMP_HALF_THICKNESS = 0.0015
_SQRT2 = math.sqrt(2.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color)


def _add_pin(
    builder: ActorBuilder,
    *,
    y: float,
    plate_half_height: float,
    pin_radius: float,
    shank_height: float,
    cone_height: float,
    tip_radius: float,
    mat: sapien.render.RenderMaterial,
) -> None:
    shank_pose = sapien.Pose(p=[0.0, y, plate_half_height + shank_height / 2.0], q=_UPRIGHT.q)
    builder.add_cylinder_collision(pose=shank_pose, radius=pin_radius, half_length=shank_height / 2.0)
    builder.add_cylinder_visual(pose=shank_pose, radius=pin_radius, half_length=shank_height / 2.0, material=mat)

    z0 = plate_half_height + shank_height
    dr = pin_radius - tip_radius
    slope = math.hypot(dr, cone_height)
    alpha = math.atan2(cone_height, dr)
    pht = _CONE_PLATE_HALF_THICKNESS
    phi = math.pi + alpha
    plate_q = [math.cos(phi / 2.0), 0.0, math.sin(phi / 2.0), 0.0]
    r_c = (pin_radius + tip_radius) / 2.0 - pht * math.sin(alpha)
    z_c = z0 + cone_height / 2.0 - pht * math.cos(alpha)
    plate_half = (slope / 2.0, pin_radius * _CONE_OVERLAP, pht)

    plate_pose_0 = sapien.Pose(p=[r_c, 0.0, z_c], q=plate_q)
    offset = sapien.Pose(p=[0.0, y, 0.0])
    for k in range(4):
        yaw = k * math.pi / 2.0
        spin = sapien.Pose(q=[math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])
        pose = offset * spin * plate_pose_0
        builder.add_box_collision(pose=pose, half_size=plate_half)
        builder.add_box_visual(pose=pose, half_size=plate_half, material=mat)


def build_pin_base(
    scene: ManiSkillScene,
    *,
    plate_half_x: float,
    plate_half_y: float,
    plate_half_height: float,
    pin_radius: float,
    shank_height: float,
    cone_height: float,
    tip_radius: float,
    pitch: float,
    plate_density: float = 1000.0,
    plate_color: tuple[float, float, float, float] = _BASE_COLOR,
    pin_color: tuple[float, float, float, float] = _PIN_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not tip_radius < pin_radius:
        raise ValueError(f"tip_radius={tip_radius} must be smaller than pin_radius={pin_radius}")
    if not 2.0 * pin_radius < pitch:
        raise ValueError(f"pitch={pitch} must exceed the pin diameter {2.0 * pin_radius}")
    if not pitch / 2.0 + pin_radius < plate_half_y:
        raise ValueError(f"plate_half_y={plate_half_y} does not carry pins at +-{pitch / 2.0}")

    builder = scene.create_actor_builder()
    plate_mat = _material(plate_color)
    pin_mat = _material(pin_color)

    plate_half = [plate_half_x, plate_half_y, plate_half_height]
    builder.add_box_collision(half_size=plate_half, density=plate_density)
    builder.add_box_visual(half_size=plate_half, material=plate_mat)

    for sign in (1.0, -1.0):
        _add_pin(
            builder,
            y=sign * pitch / 2.0,
            plate_half_height=plate_half_height,
            pin_radius=pin_radius,
            shank_height=shank_height,
            cone_height=cone_height,
            tip_radius=tip_radius,
            mat=pin_mat,
        )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def _add_square_bore(
    builder: ActorBuilder,
    *,
    y: float,
    half_width: float,
    pad_half_extent: float,
    half_thickness: float,
    chamfer: float,
    density: float,
    mat: sapien.render.RenderMaterial,
) -> None:
    s, r_out, h, c = half_width, pad_half_extent, half_thickness, chamfer

    thickness = (r_out - s) / 2.0
    offset = thickness + s
    z_c = (-h + c + h) / 2.0
    z_half = (h - (-h + c)) / 2.0
    walls = [
        ((thickness, r_out, z_half), [offset, y, z_c]),
        ((thickness, r_out, z_half), [-offset, y, z_c]),
        ((r_out, thickness, z_half), [0.0, y + offset, z_c]),
        ((r_out, thickness, z_half), [0.0, y - offset, z_c]),
    ]
    for half_size, p in walls:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=half_size, density=density)
        builder.add_box_visual(pose=pose, half_size=half_size, material=mat)

    if c <= 0.0:
        return

    half_slope = c / _SQRT2
    shift = _RAMP_HALF_THICKNESS / _SQRT2
    xy_c = s + c / 2.0 + shift
    ramp_z = -h + c / 2.0 + shift
    pt = _RAMP_HALF_THICKNESS
    cos_h, sin_h = math.cos(math.pi / 8.0), math.sin(math.pi / 8.0)
    ramps = [
        ((half_slope, r_out, pt), [xy_c, y, ramp_z], [cos_h, 0.0, sin_h, 0.0]),
        ((half_slope, r_out, pt), [-xy_c, y, ramp_z], [cos_h, 0.0, -sin_h, 0.0]),
        ((r_out, half_slope, pt), [0.0, y + xy_c, ramp_z], [cos_h, -sin_h, 0.0, 0.0]),
        ((r_out, half_slope, pt), [0.0, y - xy_c, ramp_z], [cos_h, sin_h, 0.0, 0.0]),
    ]
    for half_size, p, q in ramps:
        pose = sapien.Pose(p=p, q=q)
        builder.add_box_collision(pose=pose, half_size=half_size, density=density)
        builder.add_box_visual(pose=pose, half_size=half_size, material=mat)


def grip_bar_half_y(*, pitch: float, bore_half_width: float, chamfer: float) -> float:
    return 0.9 * (pitch / 2.0 - bore_half_width - chamfer)


def build_two_bore_bracket(
    scene: ManiSkillScene,
    *,
    bore_half_width: float,
    pad_half_extent: float,
    half_thickness: float,
    pitch: float,
    chamfer: float,
    bar_half_y: float,
    density: float = 1000.0,
    color: tuple[float, float, float, float] = _BRACKET_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not bore_half_width + chamfer < pad_half_extent:
        raise ValueError(f"bore mouth {bore_half_width + chamfer} must stay inside pad_half_extent={pad_half_extent}")
    if chamfer >= 2.0 * half_thickness:
        raise ValueError(f"chamfer={chamfer} must be shallower than the plate thickness {2.0 * half_thickness}")
    if bar_half_y < pitch / 2.0 - pad_half_extent:
        raise ValueError(
            f"bar_half_y={bar_half_y} leaves a gap between the pads (need >= {pitch / 2.0 - pad_half_extent})"
        )
    if bar_half_y > pitch / 2.0 - bore_half_width - chamfer:
        raise ValueError(
            f"bar_half_y={bar_half_y} intrudes into the bore mouths at pitch={pitch} "
            f"(need <= {pitch / 2.0 - bore_half_width - chamfer})"
        )

    builder = scene.create_actor_builder()
    mat = _material(color)

    bar_half = [pad_half_extent, bar_half_y, half_thickness]
    builder.add_box_collision(half_size=bar_half, density=density)
    builder.add_box_visual(half_size=bar_half, material=mat)

    for sign in (1.0, -1.0):
        _add_square_bore(
            builder,
            y=sign * pitch / 2.0,
            half_width=bore_half_width,
            pad_half_extent=pad_half_extent,
            half_thickness=half_thickness,
            chamfer=chamfer,
            density=density,
            mat=mat,
        )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
