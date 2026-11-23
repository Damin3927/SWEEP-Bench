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

_PAN_COLOR = (0.78, 0.80, 0.84, 1.0)
_STUB_COLOR = (0.30, 0.32, 0.36, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def scoop_chamfer_drop(chamfer_length: float, chamfer_angle: float, half_thickness: float) -> float:
    return chamfer_length * math.sin(chamfer_angle) + 2.0 * half_thickness / math.cos(chamfer_angle)


def build_scoop(
    scene: ManiSkillScene,
    *,
    inner_half_x: float,
    inner_half_y: float,
    wall_thickness: float,
    wall_height: float,
    floor_half_thickness: float,
    chamfer_length: float,
    chamfer_angle: float,
    stub_half_xy: float,
    stub_half_height: float,
    density: float = 400.0,
    pan_color: tuple[float, float, float, float] = _PAN_COLOR,
    stub_color: tuple[float, float, float, float] = _STUB_COLOR,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    pan_mat = _material(pan_color)
    ft = floor_half_thickness
    wt = wall_thickness
    slide_mat = PhysxMaterial(static_friction=0.06, dynamic_friction=0.04, restitution=0.0)
    grip_mat = PhysxMaterial(static_friction=1.0, dynamic_friction=0.9, restitution=0.0)

    floor_half = [inner_half_x + 2 * wt, inner_half_y + wt, ft]
    floor_pose = sapien.Pose(p=[0.0, wt, -ft])
    builder.add_box_collision(pose=floor_pose, half_size=floor_half, material=slide_mat, density=density)
    builder.add_box_visual(pose=floor_pose, half_size=floor_half, material=pan_mat)

    ch = chamfer_length / 2.0
    ct = ft * 0.7
    ca = chamfer_angle
    chamfer_pose = sapien.Pose(
        p=[0.0, -inner_half_y - (ch * math.cos(ca) - ct * math.sin(ca)), -(ch * math.sin(ca) + ct * math.cos(ca))],
        q=[math.cos(ca / 2.0), math.sin(ca / 2.0), 0.0, 0.0],
    )
    chamfer_half = [inner_half_x + 2 * wt, ch, ct]
    builder.add_box_collision(pose=chamfer_pose, half_size=chamfer_half, material=slide_mat, density=density)
    builder.add_box_visual(pose=chamfer_pose, half_size=chamfer_half, material=pan_mat)

    wh = wall_height / 2.0
    walls = [
        ((wt, inner_half_y + wt, wh), [inner_half_x + wt, wt, wh]),
        ((wt, inner_half_y + wt, wh), [-(inner_half_x + wt), wt, wh]),
        ((inner_half_x, wt, wh), [0.0, inner_half_y + wt, wh]),
    ]
    for half, p in walls:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=list(half), density=density)
        builder.add_box_visual(pose=pose, half_size=list(half), material=pan_mat)

    stub_pose = sapien.Pose(p=[0.0, inner_half_y + wt, wall_height + stub_half_height])
    stub_half = [stub_half_xy, stub_half_xy, stub_half_height]
    builder.add_box_collision(pose=stub_pose, half_size=stub_half, material=grip_mat, density=density)
    builder.add_box_visual(pose=stub_pose, half_size=stub_half, material=_material(stub_color))

    return builder


def build_ball(
    scene: ManiSkillScene,
    *,
    radius: float,
    color: tuple[float, float, float, float],
    density: float = 1000.0,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    material = PhysxMaterial(static_friction=0.3, dynamic_friction=0.3, restitution=0.0)
    builder.add_sphere_collision(radius=radius, material=material, density=density)
    builder.add_sphere_visual(radius=radius, material=_material(color))
    return builder
