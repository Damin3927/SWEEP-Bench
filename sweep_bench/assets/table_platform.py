from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import sapien
import sapien.render

from sweep_bench.assets.materials import flat

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.structs.actor import Actor

TOP_THICKNESS = 0.04
LEG_HALF_WIDTH = 0.03
LEG_INSET = 0.07
APRON_HEIGHT = 0.08
APRON_HALF_THICKNESS = 0.01
OVERLAY_THICKNESS = 0.002
_FRAME_RGBA = (0.16, 0.16, 0.17, 1.0)


@dataclass(frozen=True)
class FramePart:
    name: str
    center: tuple[float, float, float]
    half_size: tuple[float, float, float]


def frame_parts(half_x: float, half_y: float, height: float) -> list[FramePart]:
    under = -TOP_THICKNESS
    leg_half_z = (height - TOP_THICKNESS) / 2
    lx, ly = half_x - LEG_INSET, half_y - LEG_INSET
    parts = [
        FramePart(
            f"leg_{'p' if sx > 0 else 'n'}x_{'p' if sy > 0 else 'n'}y",
            (sx * lx, sy * ly, under - leg_half_z),
            (LEG_HALF_WIDTH, LEG_HALF_WIDTH, leg_half_z),
        )
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ]
    apron_z = under - APRON_HEIGHT / 2
    span_x, span_y = lx - LEG_HALF_WIDTH, ly - LEG_HALF_WIDTH
    for s in (-1.0, 1.0):
        parts.append(
            FramePart(
                f"apron_{'p' if s > 0 else 'n'}x",
                (s * lx, 0.0, apron_z),
                (APRON_HALF_THICKNESS, span_y, APRON_HEIGHT / 2),
            )
        )
        parts.append(
            FramePart(
                f"apron_{'p' if s > 0 else 'n'}y",
                (0.0, s * ly, apron_z),
                (span_x, APRON_HALF_THICKNESS, APRON_HEIGHT / 2),
            )
        )
    return parts


def slab_mesh(
    half_x: float, half_y: float, thickness: float, tile_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hz = thickness / 2
    center = np.array([0.0, 0.0, -hz])
    half = np.array([half_x, half_y, hz])
    vertices, normals, uvs, triangles = [], [], [], []
    for axis in range(3):
        u_axis, v_axis = [a for a in range(3) if a != axis]
        for sign in (-1.0, 1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            corners = []
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = center.copy()
                p[axis] += sign * half[axis]
                p[u_axis] += du * half[u_axis]
                p[v_axis] += dv * half[v_axis]
                corners.append(p)
            if np.dot(np.cross(corners[1] - corners[0], corners[2] - corners[0]), normal) < 0:
                corners = corners[::-1]
            base = len(vertices)
            for p in corners:
                vertices.append(p)
                normals.append(normal)
                uvs.append([p[u_axis] / tile_m, p[v_axis] / tile_m])
            triangles += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(triangles, dtype=np.uint32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32),
    )


def build_table_platform(
    scene: ManiSkillScene,
    *,
    center: tuple[float, float, float],
    half_size: tuple[float, float, float],
    top_material: sapien.render.RenderMaterial,
    tile_m: float,
    name: str,
) -> Actor:
    hx, hy, hz = half_size
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[hx, hy, hz])
    frame_material = flat(_FRAME_RGBA, roughness=0.45, metallic=0.6)
    for part in frame_parts(hx, hy, 2 * hz):
        cx, cy, cz = part.center
        pose = sapien.Pose(p=[cx, cy, cz + hz])
        builder.add_box_visual(pose=pose, half_size=list(part.half_size), material=frame_material, name=part.name)
    builder.set_initial_pose(sapien.Pose(p=list(center)))
    actor = builder.build_kinematic(name=name)
    _attach_slab(actor, hx, hy, TOP_THICKNESS, tile_m, top_material, top_z=hz)
    return actor


def build_textured_overlay(
    scene: ManiSkillScene,
    *,
    top_center: tuple[float, float, float],
    half_size: tuple[float, float],
    material: sapien.render.RenderMaterial,
    tile_m: float,
    name: str,
) -> Actor:
    builder = scene.create_actor_builder()
    x, y, z = top_center
    builder.set_initial_pose(sapien.Pose(p=[x, y, z + OVERLAY_THICKNESS]))
    actor = builder.build_kinematic(name=name)
    _attach_slab(actor, half_size[0], half_size[1], OVERLAY_THICKNESS, tile_m, material, top_z=0.0)
    return actor


def _attach_slab(
    actor: Actor,
    half_x: float,
    half_y: float,
    thickness: float,
    tile_m: float,
    material: sapien.render.RenderMaterial,
    *,
    top_z: float,
) -> None:
    vertices, triangles, normals, uvs = slab_mesh(half_x, half_y, thickness, tile_m)
    for obj in actor._objs:
        shape = sapien.render.RenderShapeTriangleMesh(
            vertices=vertices, triangles=triangles, normals=normals, uvs=uvs, material=material
        )
        shape.local_pose = sapien.Pose(p=[0.0, 0.0, top_z])
        component = sapien.render.RenderBodyComponent()
        component.attach(shape)
        obj.add_component(component)
