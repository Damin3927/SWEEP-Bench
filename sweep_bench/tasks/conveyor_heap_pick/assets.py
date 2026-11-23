from __future__ import annotations

from typing import TYPE_CHECKING

import sapien
import sapien.physx
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_CORRAL_COLOR = (0.32, 0.40, 0.48, 1.0)
_CORRAL_PAD_COLOR = (0.18, 0.18, 0.20, 1.0)


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def build_corral(
    scene: ManiSkillScene,
    *,
    inner_half_x: float,
    inner_half_y: float,
    wall_half_thickness: float,
    wall_half_height: float,
    floor_half_height: float,
    pad_half_height: float,
    pad_static_friction: float,
    pad_dynamic_friction: float,
    color: tuple[float, float, float, float] = _CORRAL_COLOR,
    pad_color: tuple[float, float, float, float] = _CORRAL_PAD_COLOR,
) -> ActorBuilder:
    if not 0.0 < pad_half_height < floor_half_height:
        raise ValueError(f"pad_half_height={pad_half_height} must lie strictly inside floor_half_height")
    ihx, ihy, wht = inner_half_x, inner_half_y, wall_half_thickness
    whh, fhh = wall_half_height, floor_half_height
    outer_half_x = ihx + 2.0 * wht
    outer_half_y = ihy + 2.0 * wht
    mat = _material(color)

    builder = scene.create_actor_builder()

    def add_box(
        half: list[float],
        pose: sapien.Pose | None = None,
        *,
        material: sapien.render.RenderMaterial | None = None,
        physical: sapien.physx.PhysxMaterial | None = None,
    ) -> None:
        kwargs = {} if pose is None else {"pose": pose}
        builder.add_box_collision(half_size=half, material=physical, **kwargs)
        builder.add_box_visual(half_size=half, material=material if material is not None else mat, **kwargs)

    body_half_z = fhh - pad_half_height
    add_box([outer_half_x, outer_half_y, body_half_z], sapien.Pose(p=[0.0, 0.0, -pad_half_height]))
    add_box(
        [outer_half_x, outer_half_y, pad_half_height],
        sapien.Pose(p=[0.0, 0.0, fhh - pad_half_height]),
        material=_material(pad_color),
        physical=sapien.physx.PhysxMaterial(
            static_friction=pad_static_friction,
            dynamic_friction=pad_dynamic_friction,
            restitution=0.0,
        ),
    )

    wall_z = fhh + whh
    for side in (1.0, -1.0):
        add_box([wht, outer_half_y, whh], sapien.Pose(p=[side * (ihx + wht), 0.0, wall_z]))
    for side in (1.0, -1.0):
        add_box([ihx, wht, whh], sapien.Pose(p=[0.0, side * (ihy + wht), wall_z]))

    return builder
