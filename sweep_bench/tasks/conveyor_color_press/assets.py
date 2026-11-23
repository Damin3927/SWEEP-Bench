from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_PANEL_COLOR = (0.42, 0.44, 0.48, 1.0)

CAP_PALETTE: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("red", (0.87, 0.18, 0.15, 1.0)),
    ("green", (0.16, 0.68, 0.24, 1.0)),
    ("blue", (0.14, 0.34, 0.85, 1.0)),
    ("yellow", (0.94, 0.82, 0.12, 1.0)),
)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def build_button_panel(
    scene: ManiSkillScene,
    *,
    bore_half_width: float,
    bore_pitch: float,
    num_bores: int,
    wall_thickness: float,
    wall_height: float,
    floor_half_thickness: float,
    color: tuple[float, float, float, float] = _PANEL_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    s, pitch, w, hw, tf = bore_half_width, bore_pitch, wall_thickness, wall_height, floor_half_thickness
    if num_bores < 2:
        raise ValueError(f"num_bores={num_bores} must be at least 2 (a one-bore panel is the press-button task)")
    divider = pitch - 2.0 * s
    if divider <= 0.0:
        raise ValueError(f"bore_pitch={pitch} leaves no divider between bores of half-width {s}")
    mat = _material(color)

    r_x = 0.5 * (num_bores - 1) * pitch + s + w
    r_y = s + w

    builder = scene.create_actor_builder()
    floor_pose = sapien.Pose(p=[0.0, 0.0, -tf])
    builder.add_box_collision(pose=floor_pose, half_size=[r_x, r_y, tf])
    builder.add_box_visual(pose=floor_pose, half_size=[r_x, r_y, tf], material=mat)

    walls: list[tuple[tuple[float, float, float], list[float]]] = [
        ((0.5 * w, r_y, hw * 0.5), [r_x - 0.5 * w, 0.0, hw * 0.5]),
        ((0.5 * w, r_y, hw * 0.5), [-(r_x - 0.5 * w), 0.0, hw * 0.5]),
        ((r_x, 0.5 * w, hw * 0.5), [0.0, r_y - 0.5 * w, hw * 0.5]),
        ((r_x, 0.5 * w, hw * 0.5), [0.0, -(r_y - 0.5 * w), hw * 0.5]),
    ]
    for k in range(num_bores - 1):
        centre_x = (k - 0.5 * (num_bores - 1) + 0.5) * pitch
        walls.append(((0.5 * divider, r_y, hw * 0.5), [centre_x, 0.0, hw * 0.5]))
    for half_size, p in walls:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=list(half_size))
        builder.add_box_visual(pose=pose, half_size=list(half_size), material=mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_button_cap(
    scene: ManiSkillScene,
    *,
    cap_half_width: float,
    cap_half_height: float,
    color: tuple[float, float, float, float],
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    aw, ah = cap_half_width, cap_half_height
    body = (color[0] * 0.55, color[1] * 0.55, color[2] * 0.55, color[3])
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[aw, aw, ah])
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, ah * 0.5]),
        half_size=[aw, aw, ah * 0.5],
        material=_material(color),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, -ah * 0.5]),
        half_size=[aw, aw, ah * 0.5],
        material=_material(body),
    )
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
