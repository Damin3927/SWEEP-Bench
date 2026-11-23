from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_CARRIER_COLOR = (0.42, 0.44, 0.48, 1.0)
_SLIDER_BASE_COLOR = (0.22, 0.24, 0.28, 1.0)
_SLIDER_POST_COLOR = (0.93, 0.93, 0.90, 1.0)

WELL_NAMES: tuple[str, str, str] = ("red", "green", "blue")
WELL_PATCH_COLORS: tuple[tuple[float, float, float, float], ...] = (
    (0.85, 0.16, 0.16, 1.0),
    (0.16, 0.70, 0.28, 1.0),
    (0.18, 0.36, 0.88, 1.0),
)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def build_selector_carrier(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    channel_half_x: float,
    channel_half_y: float,
    channel_height: float,
    floor_half_thickness: float,
    top_half_thickness: float,
    slot_half_x: float,
    density: float,
    friction: float,
    well_offsets: Sequence[float],
    patch_half: tuple[float, float, float] = (0.007, 0.008, 0.0012),
    color: tuple[float, float, float, float] = _CARRIER_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if not channel_half_x < half_x:
        raise ValueError(f"channel_half_x={channel_half_x} must be smaller than half_x={half_x}")
    if not slot_half_x < channel_half_x:
        raise ValueError(f"slot_half_x={slot_half_x} must be smaller than channel_half_x={channel_half_x}")
    if not channel_half_y < half_y:
        raise ValueError(f"channel_half_y={channel_half_y} must be smaller than half_y={half_y}")
    if len(well_offsets) != len(WELL_PATCH_COLORS):
        raise ValueError(f"expected {len(WELL_PATCH_COLORS)} well offsets, got {len(well_offsets)}")

    mat = _material(color)
    phys = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    builder = scene.create_actor_builder()

    def box(pose: sapien.Pose, half: list[float], visual: sapien.render.RenderMaterial = mat) -> None:
        builder.add_box_collision(pose=pose, half_size=half, density=density, material=phys)
        builder.add_box_visual(pose=pose, half_size=half, material=visual)

    box(sapien.Pose(p=[0.0, 0.0, -floor_half_thickness]), [half_x, half_y, floor_half_thickness])

    wall_half_x = (half_x - channel_half_x) * 0.5
    for sx in (-1.0, 1.0):
        box(
            sapien.Pose(p=[sx * (channel_half_x + wall_half_x), 0.0, channel_height * 0.5]),
            [wall_half_x, half_y, channel_height * 0.5],
        )

    end_half_y = (half_y - channel_half_y) * 0.5
    for sy in (-1.0, 1.0):
        box(
            sapien.Pose(p=[0.0, sy * (channel_half_y + end_half_y), channel_height * 0.5]),
            [channel_half_x, end_half_y, channel_height * 0.5],
        )

    top_half_x = (half_x - slot_half_x) * 0.5
    for sx in (-1.0, 1.0):
        box(
            sapien.Pose(p=[sx * (slot_half_x + top_half_x), 0.0, channel_height + top_half_thickness]),
            [top_half_x, half_y, top_half_thickness],
        )

    patch_z = channel_height + 2.0 * top_half_thickness + patch_half[2]
    for sx in (-1.0, 1.0):
        for well_y, patch_color in zip(well_offsets, WELL_PATCH_COLORS, strict=True):
            builder.add_box_visual(
                pose=sapien.Pose(p=[sx * (slot_half_x + top_half_x), float(well_y), patch_z]),
                half_size=list(patch_half),
                material=_material(patch_color),
            )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_selector_slider(
    scene: ManiSkillScene,
    *,
    base_half: tuple[float, float, float],
    post_half: tuple[float, float, float],
    base_density: float,
    post_density: float,
    friction: float,
    base_color: tuple[float, float, float, float] = _SLIDER_BASE_COLOR,
    post_color: tuple[float, float, float, float] = _SLIDER_POST_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    phys = sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    builder = scene.create_actor_builder()

    builder.add_box_collision(half_size=list(base_half), density=base_density, material=phys)
    builder.add_box_visual(half_size=list(base_half), material=_material(base_color))

    post_pose = sapien.Pose(p=[0.0, 0.0, base_half[2] + post_half[2]])
    builder.add_box_collision(pose=post_pose, half_size=list(post_half), density=post_density, material=phys)
    builder.add_box_visual(pose=post_pose, half_size=list(post_half), material=_material(post_color))

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
