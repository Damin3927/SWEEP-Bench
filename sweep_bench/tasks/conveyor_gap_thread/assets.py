from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_CRATE_COLOR = (0.34, 0.38, 0.46, 1.0)
_DIVIDER_COLOR = (0.85, 0.62, 0.18, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def build_grooved_crate(
    scene: ManiSkillScene,
    *,
    base_half_x: float,
    base_half_y: float,
    base_half_z: float,
    pad_half_z: float,
    groove_center_y: float,
    groove_half_width: float,
    wall_half_thickness: float,
    wall_half_height: float,
    color: tuple[float, float, float, float] = _CRATE_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if groove_center_y + groove_half_width >= base_half_y:
        raise ValueError(
            f"groove at |y| = {groove_center_y:.4f} +- {groove_half_width:.4f} does not fit inside a "
            f"{base_half_y:.4f} m base half-width"
        )
    mat = _material(color)
    builder = scene.create_actor_builder()

    def add_box(half: list[float], pose: sapien.Pose | None = None) -> None:
        kwargs = {} if pose is None else {"pose": pose}
        builder.add_box_collision(half_size=half, **kwargs)
        builder.add_box_visual(half_size=half, material=mat, **kwargs)

    add_box([base_half_x, base_half_y, base_half_z])

    pad_z = base_half_z + pad_half_z
    floor_top = base_half_z + 2.0 * pad_half_z

    centre_half_y = groove_center_y - groove_half_width
    add_box([base_half_x, centre_half_y, pad_half_z], sapien.Pose(p=[0.0, 0.0, pad_z]))

    outer_inner = groove_center_y + groove_half_width
    outer_half_y = (base_half_y - outer_inner) / 2.0
    for side in (1.0, -1.0):
        add_box(
            [base_half_x, outer_half_y, pad_half_z],
            sapien.Pose(p=[0.0, side * (outer_inner + outer_half_y), pad_z]),
        )

    for side in (1.0, -1.0):
        add_box(
            [base_half_x, wall_half_thickness, wall_half_height],
            sapien.Pose(p=[0.0, side * (base_half_y - wall_half_thickness), floor_top + wall_half_height]),
        )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_divider(
    scene: ManiSkillScene,
    *,
    half_length: float,
    half_thickness: float,
    half_height: float,
    density: float,
    color: tuple[float, float, float, float] = _DIVIDER_COLOR,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    half = [half_length, half_thickness, half_height]
    builder.add_box_collision(half_size=half, density=density)
    builder.add_box_visual(half_size=half, material=_material(color))
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
