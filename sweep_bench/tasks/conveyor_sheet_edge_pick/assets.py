from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_RACK_COLOR = (0.38, 0.40, 0.45, 1.0)
_FRAME_COLOR = (0.85, 0.55, 0.15, 1.0)
_PANE_COLOR = (0.35, 0.62, 0.72, 1.0)

RACK_FRICTION = 0.7
FRAME_FRICTION = 0.8
PANE_FRICTION = 0.5


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6, metallic=0.1)


def _add_box(
    builder: ActorBuilder,
    center: tuple[float, float, float],
    half: tuple[float, float, float],
    color: tuple[float, float, float, float],
    density: float,
    material: sapien.physx.PhysxMaterial,
) -> None:
    pose = sapien.Pose(p=list(center))
    builder.add_box_collision(pose=pose, half_size=half, density=density, material=material)
    builder.add_box_visual(pose=pose, half_size=half, material=_material(color))


def build_sheet_rack(
    scene: ManiSkillScene,
    *,
    plate_half_x: float,
    plate_half_y: float,
    plate_half_thickness: float,
    rib_half_x: float,
    rib_half_y: float,
    rib_height: float,
    rib_y: float,
    density: float,
    friction: float = RACK_FRICTION,
    color: tuple[float, float, float, float] = _RACK_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not rib_half_y < rib_y:
        raise ValueError(f"rib_half_y={rib_half_y} must be smaller than rib_y={rib_y} (the ribs would merge)")
    if rib_y + rib_half_y > plate_half_y:
        raise ValueError(f"ribs at |y|={rib_y}+{rib_half_y} overhang the plate half-length {plate_half_y}")
    if rib_height <= 0.0:
        raise ValueError(f"rib_height={rib_height} must be positive")

    phys = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    builder = scene.create_actor_builder()

    _add_box(
        builder,
        (0.0, 0.0, -plate_half_thickness),
        (plate_half_x, plate_half_y, plate_half_thickness),
        color,
        density,
        phys,
    )
    for sign in (-1.0, 1.0):
        _add_box(
            builder,
            (0.0, sign * rib_y, 0.5 * rib_height),
            (rib_half_x, rib_half_y, 0.5 * rib_height),
            color,
            density,
            phys,
        )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_framed_sheet(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    rail_width_x: float,
    rail_width_y: float,
    rail_thickness: float,
    ledge_width: float,
    ledge_thickness: float,
    density: float,
    friction: float = FRAME_FRICTION,
    color: tuple[float, float, float, float] = _FRAME_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if rail_width_x <= 0.0 or rail_width_y <= 0.0:
        raise ValueError("rail widths must be positive")
    if rail_width_x >= half_x or rail_width_y >= half_y:
        raise ValueError("rail widths must leave an aperture inside the frame")
    if not 0.0 < ledge_thickness < rail_thickness:
        raise ValueError(f"ledge_thickness={ledge_thickness} must sit inside rail_thickness={rail_thickness}")
    inner_x = half_x - rail_width_x
    inner_y = half_y - rail_width_y
    if ledge_width >= inner_x or ledge_width >= inner_y:
        raise ValueError(f"ledge_width={ledge_width} would close the frame's aperture")

    phys = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    builder = scene.create_actor_builder()

    half_t = 0.5 * rail_thickness
    for sign in (-1.0, 1.0):
        _add_box(
            builder,
            (sign * (half_x - 0.5 * rail_width_x), 0.0, half_t),
            (0.5 * rail_width_x, half_y, half_t),
            color,
            density,
            phys,
        )
        _add_box(
            builder,
            (0.0, sign * (half_y - 0.5 * rail_width_y), half_t),
            (inner_x, 0.5 * rail_width_y, half_t),
            color,
            density,
            phys,
        )
        _add_box(
            builder,
            (sign * (inner_x - 0.5 * ledge_width), 0.0, 0.5 * ledge_thickness),
            (0.5 * ledge_width, inner_y, 0.5 * ledge_thickness),
            color,
            density,
            phys,
        )
        _add_box(
            builder,
            (0.0, sign * (inner_y - 0.5 * ledge_width), 0.5 * ledge_thickness),
            (inner_x - ledge_width, 0.5 * ledge_width, 0.5 * ledge_thickness),
            color,
            density,
            phys,
        )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_pane(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_thickness: float,
    density: float,
    friction: float = PANE_FRICTION,
    color: tuple[float, float, float, float] = _PANE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    phys = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    builder = scene.create_actor_builder()
    _add_box(builder, (0.0, 0.0, 0.0), (half_x, half_y, half_thickness), color, density, phys)
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
