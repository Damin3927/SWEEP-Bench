from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder

_PALLET_COLOR = (0.34, 0.38, 0.46, 1.0)
_ARCH_COLOR = (0.85, 0.55, 0.10, 1.0)

_CHAMFER_HALF_THICKNESS = 0.0015


def _material(color: tuple[float, float, float, float]) -> sapien.render.RenderMaterial:
    return flat(color, roughness=0.6)


def build_arch_pallet(
    scene: ManiSkillScene,
    *,
    bin_inner_half: float,
    bin_wall_half_thickness: float,
    bin_wall_half_height: float,
    plate_half_thickness: float,
    plate_half_x: float,
    plate_back_y: float,
    plate_front_y: float,
    arch_offset: float,
    arch_clear_height: float,
    rail_half_x: float,
    rail_half_y: float,
    rail_half_z: float,
    upright_half_x: float,
    upright_half_y: float,
    upright_local_x: float,
    chamfer: float = 0.0,
    arch_density: float = 300.0,
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    if plate_front_y <= plate_back_y:
        raise ValueError(f"plate_front_y ({plate_front_y}) must exceed plate_back_y ({plate_back_y})")
    if arch_offset <= bin_inner_half + 2.0 * bin_wall_half_thickness + rail_half_y:
        raise ValueError(
            f"arch_offset ({arch_offset}) must clear the bin's own footprint so the corridor is a separate place"
        )
    bih, bwt, bwh, pht = bin_inner_half, bin_wall_half_thickness, bin_wall_half_height, plate_half_thickness
    plate_top = pht
    wall_z = plate_top + bwh
    rim_z = plate_top + 2.0 * bwh
    pallet_mat = _material(_PALLET_COLOR)
    arch_mat = _material(_ARCH_COLOR)

    builder = scene.create_actor_builder()

    def add_box(
        half: list[float],
        pose: sapien.Pose | None = None,
        *,
        material: sapien.render.RenderMaterial = pallet_mat,
        density: float = 1000.0,
    ) -> None:
        kwargs = {} if pose is None else {"pose": pose}
        builder.add_box_collision(half_size=half, density=density, **kwargs)
        builder.add_box_visual(half_size=half, material=material, **kwargs)

    plate_half_y = (plate_front_y - plate_back_y) / 2.0
    plate_centre_y = (plate_front_y + plate_back_y) / 2.0
    add_box([plate_half_x, plate_half_y, pht], sapien.Pose(p=[0.0, plate_centre_y, 0.0]))

    bin_outer_half = bih + 2.0 * bwt
    for side in (1.0, -1.0):
        add_box([bwt, bin_outer_half, bwh], sapien.Pose(p=[side * (bih + bwt), 0.0, wall_z]))
        add_box([bih, bwt, bwh], sapien.Pose(p=[0.0, side * (bih + bwt), wall_z]))

    if chamfer > 0.0:
        flare_half = chamfer * math.sqrt(2) / 2.0
        for side in (1.0, -1.0):
            theta = math.pi / 2 - side * math.pi / 4
            add_box(
                [bih, flare_half, _CHAMFER_HALF_THICKNESS],
                sapien.Pose(
                    p=[0.0, side * (bih + chamfer / 2.0), rim_z + chamfer / 2.0],
                    q=[math.cos(theta / 2), math.sin(theta / 2), 0.0, 0.0],
                ),
            )
            phi = -(math.pi / 2 - side * math.pi / 4)
            add_box(
                [flare_half, bin_outer_half, _CHAMFER_HALF_THICKNESS],
                sapien.Pose(
                    p=[side * (bih + chamfer / 2.0), 0.0, rim_z + chamfer / 2.0],
                    q=[math.cos(phi / 2), 0.0, math.sin(phi / 2), 0.0],
                ),
            )

    upright_half_z = arch_clear_height / 2.0
    add_box(
        [upright_half_x, upright_half_y, upright_half_z],
        sapien.Pose(p=[upright_local_x, -arch_offset, plate_top + upright_half_z]),
        material=arch_mat,
        density=arch_density,
    )
    add_box(
        [rail_half_x, rail_half_y, rail_half_z],
        sapien.Pose(p=[0.0, -arch_offset, plate_top + arch_clear_height + rail_half_z]),
        material=arch_mat,
        density=arch_density,
    )

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder
