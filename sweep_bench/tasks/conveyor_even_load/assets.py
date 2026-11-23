from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_HOUSING_COLOR = (0.38, 0.40, 0.44, 1.0)
_PAD_A_COLOR = (0.85, 0.55, 0.15, 1.0)
_PAD_B_COLOR = (0.25, 0.60, 0.85, 1.0)

PAD_DENSITY = 1000.0

HOUSING_FRICTION = 0.05


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, metallic=0.2)


def build_twin_pad_carrier(
    scene: ManiSkillScene,
    *,
    bore_half_width: float,
    pad_spacing: float,
    outer_half_x: float,
    outer_half_y: float,
    wall_height: float,
    floor_half_thickness: float,
    friction: float = HOUSING_FRICTION,
    color: tuple[float, float, float, float] = _HOUSING_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    s_bore, rx, ry, hw, tf = bore_half_width, outer_half_x, outer_half_y, wall_height, floor_half_thickness
    half_s = 0.5 * pad_spacing
    if not s_bore < rx:
        raise ValueError(f"bore_half_width={s_bore} must be smaller than outer_half_x={rx}")
    if not s_bore < half_s:
        raise ValueError(f"bore_half_width={s_bore} must be smaller than half the pad_spacing={half_s} (bores overlap)")
    if not half_s + s_bore < ry:
        raise ValueError(
            f"outer_half_y={ry} leaves no wall outside the bores at +-{half_s} (needs > {half_s + s_bore})"
        )

    mat = _material(color)
    phys = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    builder = scene.create_actor_builder()

    floor_pose = sapien.Pose(p=[0.0, 0.0, -tf])
    builder.add_box_collision(pose=floor_pose, half_size=[rx, ry, tf], material=phys)
    builder.add_box_visual(pose=floor_pose, half_size=[rx, ry, tf], material=mat)

    t_x = 0.5 * (rx - s_bore)
    t_out = 0.5 * (ry - half_s - s_bore)
    walls = [
        ((t_x, ry, hw * 0.5), [s_bore + t_x, 0.0, hw * 0.5]),
        ((t_x, ry, hw * 0.5), [-(s_bore + t_x), 0.0, hw * 0.5]),
        ((rx, half_s - s_bore, hw * 0.5), [0.0, 0.0, hw * 0.5]),
        ((rx, t_out, hw * 0.5), [0.0, half_s + s_bore + t_out, hw * 0.5]),
        ((rx, t_out, hw * 0.5), [0.0, -(half_s + s_bore + t_out), hw * 0.5]),
    ]
    for half_size, p in walls:
        pose = sapien.Pose(p=list(p))
        builder.add_box_collision(pose=pose, half_size=list(half_size), material=phys)
        builder.add_box_visual(pose=pose, half_size=list(half_size), material=mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_load_pad(
    scene: ManiSkillScene,
    *,
    half_width: float,
    half_thickness: float,
    density: float = PAD_DENSITY,
    color: tuple[float, float, float, float] = _PAD_A_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if half_width <= 0.0 or half_thickness <= 0.0:
        raise ValueError(f"pad half extents must be positive, got {half_width} x {half_thickness}")
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[half_width, half_width, half_thickness], density=density)
    builder.add_box_visual(half_size=[half_width, half_width, half_thickness], material=_material(color))
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


PAD_COLORS: tuple[tuple[float, float, float, float], tuple[float, float, float, float]] = (_PAD_A_COLOR, _PAD_B_COLOR)
