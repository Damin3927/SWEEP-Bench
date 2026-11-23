from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_BODY_COLOR = (0.55, 0.57, 0.60, 1.0)
_KNOB_COLOR = (0.90, 0.55, 0.10, 1.0)

_UPRIGHT = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def build_plug_lid(
    scene: ManiSkillScene,
    *,
    knob_radius: float,
    knob_half_height: float,
    cap_radius: float,
    cap_half_thickness: float,
    plug_radius: float,
    plug_half_height: float,
    body_color: tuple[float, float, float, float] = _BODY_COLOR,
    knob_color: tuple[float, float, float, float] = _KNOB_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    knob_mat = _material(knob_color)
    body_mat = _material(body_color)
    builder = scene.create_actor_builder()

    builder.add_cylinder_collision(pose=_UPRIGHT, radius=knob_radius, half_length=knob_half_height)
    builder.add_cylinder_visual(pose=_UPRIGHT, radius=knob_radius, half_length=knob_half_height, material=knob_mat)

    cap_z = -(knob_half_height + cap_half_thickness)
    cap_pose = sapien.Pose(p=[0.0, 0.0, cap_z], q=_UPRIGHT.q)
    builder.add_cylinder_collision(pose=cap_pose, radius=cap_radius, half_length=cap_half_thickness)
    builder.add_cylinder_visual(pose=cap_pose, radius=cap_radius, half_length=cap_half_thickness, material=body_mat)

    plug_z = -(knob_half_height + 2.0 * cap_half_thickness + plug_half_height)
    plug_pose = sapien.Pose(p=[0.0, 0.0, plug_z], q=_UPRIGHT.q)
    builder.add_cylinder_collision(pose=plug_pose, radius=plug_radius, half_length=plug_half_height)
    builder.add_cylinder_visual(pose=plug_pose, radius=plug_radius, half_length=plug_half_height, material=body_mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
