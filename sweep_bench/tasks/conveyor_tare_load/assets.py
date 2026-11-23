from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_WELL_COLOR = (0.38, 0.40, 0.44, 1.0)
_PAN_COLOR = (0.78, 0.80, 0.84, 1.0)
_RIM_COLOR = (0.90, 0.62, 0.15, 1.0)
_PLUNGER_BODY_COLOR = (0.30, 0.32, 0.36, 1.0)
_PLUNGER_FACE_COLOR = (0.87, 0.24, 0.20, 1.0)

PAN_DENSITY = 200.0
PLUNGER_DENSITY = 250.0

WELL_FRICTION = 0.05


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, metallic=0.2)


def build_scale_well(
    scene: ManiSkillScene,
    *,
    bore_half_width: float,
    outer_half_width: float,
    wall_height: float,
    floor_half_thickness: float,
    friction: float = WELL_FRICTION,
    color: tuple[float, float, float, float] = _WELL_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    s, r_out, hw, tf = bore_half_width, outer_half_width, wall_height, floor_half_thickness
    if not s < r_out:
        raise ValueError(f"bore_half_width={s} must be smaller than outer_half_width={r_out}")
    mat = _material(color)
    phys = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)

    builder = scene.create_actor_builder()
    floor_pose = sapien.Pose(p=[0.0, 0.0, -tf])
    builder.add_box_collision(pose=floor_pose, half_size=[r_out, r_out, tf], material=phys)
    builder.add_box_visual(pose=floor_pose, half_size=[r_out, r_out, tf], material=mat)

    thickness = (r_out - s) * 0.5
    offset = thickness + s
    walls = [
        ((thickness, r_out, hw * 0.5), [offset, 0.0, hw * 0.5]),
        ((thickness, r_out, hw * 0.5), [-offset, 0.0, hw * 0.5]),
        ((r_out, thickness, hw * 0.5), [0.0, offset, hw * 0.5]),
        ((r_out, thickness, hw * 0.5), [0.0, -offset, hw * 0.5]),
    ]
    for half_size, p in walls:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=half_size, material=phys)
        builder.add_box_visual(pose=pose, half_size=half_size, material=mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_scale_pan(
    scene: ManiSkillScene,
    *,
    plate_half_width: float,
    plate_half_thickness: float,
    rim_half_thickness: float,
    rim_height: float,
    density: float = PAN_DENSITY,
    plate_color: tuple[float, float, float, float] = _PAN_COLOR,
    rim_color: tuple[float, float, float, float] = _RIM_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    phw, pht, rt, rh = plate_half_width, plate_half_thickness, rim_half_thickness, rim_height
    if not 2.0 * rt < phw:
        raise ValueError(f"rim_half_thickness={rt} leaves no clear interior inside plate_half_width={phw}")
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[phw, phw, pht], density=density)
    builder.add_box_visual(half_size=[phw, phw, pht], material=_material(plate_color))

    rim_mat = _material(rim_color)
    rim_center_z = pht + rh * 0.5
    offset = phw - rt
    rims = [
        ((rt, phw, rh * 0.5), [offset, 0.0, rim_center_z]),
        ((rt, phw, rh * 0.5), [-offset, 0.0, rim_center_z]),
        ((phw, rt, rh * 0.5), [0.0, offset, rim_center_z]),
        ((phw, rt, rh * 0.5), [0.0, -offset, rim_center_z]),
    ]
    for half_size, p in rims:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=half_size, density=density)
        builder.add_box_visual(pose=pose, half_size=half_size, material=rim_mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_tare_plunger(
    scene: ManiSkillScene,
    *,
    half_width: float,
    half_height: float,
    face_half_height: float = 0.003,
    density: float = PLUNGER_DENSITY,
    body_color: tuple[float, float, float, float] = _PLUNGER_BODY_COLOR,
    face_color: tuple[float, float, float, float] = _PLUNGER_FACE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not face_half_height < half_height:
        raise ValueError(f"face_half_height={face_half_height} must be thinner than half_height={half_height}")
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[half_width, half_width, half_height], density=density)
    body_half_h = half_height - face_half_height
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, -face_half_height]),
        half_size=[half_width, half_width, body_half_h],
        material=_material(body_color),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, half_height - face_half_height]),
        half_size=[half_width, half_width, face_half_height],
        material=_material(face_color),
    )
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
