from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.physx
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_SLEEVE_COLOR = (0.24, 0.28, 0.36, 1.0)
_TRAY_COLOR = (0.88, 0.80, 0.52, 1.0)
_LIP_COLOR = (0.78, 0.32, 0.22, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.7)


def _box(
    builder: ActorBuilder,
    center: tuple[float, float, float],
    half: tuple[float, float, float],
    color: tuple[float, float, float, float],
    density: float,
    material: sapien.physx.PhysxMaterial | None = None,
) -> None:
    pose = sapien.Pose(p=list(center))
    builder.add_box_collision(pose=pose, half_size=half, density=density, material=material)
    builder.add_box_visual(pose=pose, half_size=half, material=_material(color))


def build_drawer_sleeve(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    height: float,
    floor_top: float,
    wall_inner_y: float,
    lip_inner_y: float,
    lip_bottom: float,
    front_stop_x: float,
    front_stop_inner_y: float,
    rear_stop_x: float,
    rear_stop_top: float,
    tray_top: float,
    engage_dx: float,
    density: float,
) -> ActorBuilder:
    if not floor_top < tray_top < lip_bottom < height:
        raise ValueError(f"bad z stack: floor_top={floor_top} tray_top={tray_top} lip={lip_bottom} height={height}")
    if not lip_inner_y < wall_inner_y < half_y:
        raise ValueError(f"bad y stack: lip_inner_y={lip_inner_y} wall_inner_y={wall_inner_y} half_y={half_y}")

    builder = scene.create_actor_builder()
    origin_z = 0.5 * height

    def place(cx: float, cy: float, z_lo: float, z_hi: float, hx: float, hy: float, color) -> None:
        _box(
            builder,
            (cx - engage_dx, cy, 0.5 * (z_lo + z_hi) - origin_z),
            (hx, hy, 0.5 * (z_hi - z_lo)),
            color,
            density,
        )

    place(0.0, 0.0, 0.0, floor_top, half_x, half_y, _SLEEVE_COLOR)
    wall_half_y = 0.5 * (half_y - wall_inner_y)
    lip_half_y = 0.5 * (wall_inner_y - lip_inner_y)
    for s in (-1.0, 1.0):
        place(0.0, s * (half_y + wall_inner_y) * 0.5, floor_top, height, half_x, wall_half_y, _SLEEVE_COLOR)
        place(0.0, s * (wall_inner_y + lip_inner_y) * 0.5, lip_bottom, height, half_x, lip_half_y, _SLEEVE_COLOR)
        place(
            0.5 * (front_stop_x - half_x),
            s * (wall_inner_y + front_stop_inner_y) * 0.5,
            floor_top,
            tray_top,
            0.5 * (front_stop_x + half_x),
            0.5 * (wall_inner_y - front_stop_inner_y),
            _SLEEVE_COLOR,
        )
    place(
        0.5 * (rear_stop_x + half_x),
        0.0,
        floor_top,
        rear_stop_top,
        0.5 * (half_x - rear_stop_x),
        wall_inner_y,
        _SLEEVE_COLOR,
    )
    return builder


def build_drawer_tray(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    thickness: float,
    ear_front_x: float,
    ear_outer_y: float,
    lip_x0: float,
    lip_x1: float,
    lip_half_y: float,
    lip_height: float,
    density: float,
    friction: float,
) -> ActorBuilder:
    if not lip_x0 < lip_x1 <= half_x:
        raise ValueError(f"lip span [{lip_x0}, {lip_x1}] must start ahead of and end within the body (+-{half_x})")
    if not -half_x < ear_front_x < half_x:
        raise ValueError(f"ear_front_x={ear_front_x} must lie inside the body (+-{half_x})")
    if ear_outer_y <= half_y:
        raise ValueError(f"ear_outer_y={ear_outer_y} must reach beyond the body half-width {half_y}")

    builder = scene.create_actor_builder()
    mat = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    half_z = 0.5 * thickness

    _box(builder, (0.0, 0.0, 0.0), (half_x, half_y, half_z), _TRAY_COLOR, density, mat)
    ear_half_x = 0.5 * (half_x - ear_front_x)
    ear_half_y = 0.5 * (ear_outer_y - half_y)
    for s in (-1.0, 1.0):
        _box(
            builder,
            (0.5 * (ear_front_x + half_x), s * 0.5 * (ear_outer_y + half_y), 0.0),
            (ear_half_x, ear_half_y, half_z),
            _TRAY_COLOR,
            density,
            mat,
        )
    _box(
        builder,
        (0.5 * (lip_x0 + lip_x1), 0.0, half_z + 0.5 * lip_height),
        (0.5 * (lip_x1 - lip_x0), lip_half_y, 0.5 * lip_height),
        _LIP_COLOR,
        density,
        mat,
    )
    return builder
