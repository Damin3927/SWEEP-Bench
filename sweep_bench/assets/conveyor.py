from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import sapien
import sapien.physx
import sapien.render

from sweep_bench.assets.conveyor_drive import BeltDrive, apply_drive_writes
from sweep_bench.assets.materials import flat, normal_array, rgba_array, rgba_texture, smooth_wrap

if TYPE_CHECKING:
    import torch
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.structs.actor import Actor

logger = logging.getLogger(__name__)

_ROLLER_RADIUS = 0.03
_RAIL_HALF_THICKNESS = 0.0075
_RAIL_LIP = 0.002
_LEG_HALF_XSECTION = 0.012
_LEG_MAX_SPACING = 0.7

_BELT_TEXTURE_PERIOD_M = 0.2
_TEXTURE_RES = 256
_TEXTURE_SEED = 0xBE17

_VERTEX_V_COL = 7


def _belt_height_field(rng: np.random.Generator) -> np.ndarray:
    res = _TEXTURE_RES
    grain = smooth_wrap(rng.random((res, res)), passes=2)
    u = np.arange(res)[None, :] / res
    v = np.arange(res)[:, None] / res
    chevron_phase = (v * 2.0 + np.abs(u - 0.5)) % 1.0
    ridge = np.exp(-((chevron_phase - 0.5) ** 2) / (2 * 0.04**2))
    seam_dist = np.minimum(v, 1.0 - v) + np.zeros_like(u)
    seam = np.exp(-(seam_dist**2) / (2 * 0.012**2))
    return np.clip(0.35 * grain + 0.65 * ridge - 0.8 * seam, -1.0, 1.0)


@dataclass
class _BeltTextures:
    textures: list[sapien.render.RenderTexture2D]
    arrays: list[np.ndarray]


def _make_belt_material(rng: np.random.Generator) -> tuple[sapien.render.RenderMaterial, _BeltTextures]:
    height = _belt_height_field(rng)
    shade = np.clip(0.07 + 0.09 * height, 0.02, 1.0)[..., None] * np.array([1.0, 1.0, 1.08])
    roughness = np.clip(0.88 - 0.22 * height, 0.0, 1.0)

    arrays = [
        rgba_array(shade),
        rgba_array(np.repeat(roughness[..., None], 3, axis=-1)),
        normal_array(height, strength=3.0),
    ]
    textures = [rgba_texture(a) for a in arrays]

    mat = sapien.render.RenderMaterial()
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    mat.roughness = 1.0
    mat.metallic = 0.0
    mat.base_color_texture = textures[0]
    mat.roughness_texture = textures[1]
    mat.normal_texture = textures[2]
    return mat, _BeltTextures(textures=textures, arrays=arrays)


def _make_steel_material(rng: np.random.Generator) -> sapien.render.RenderMaterial:
    res = _TEXTURE_RES
    streaks = rng.random((res, res))
    for _ in range(3):
        acc = np.zeros_like(streaks)
        for d in (-2, -1, 0, 1, 2):
            acc += np.roll(streaks, d, axis=0)
        streaks = acc / 5.0
    streaks = smooth_wrap(streaks, passes=1)
    shade = (0.55 + 0.18 * streaks)[..., None] * np.array([0.96, 0.98, 1.0])
    roughness = 0.40 + 0.18 * streaks

    mat = sapien.render.RenderMaterial()
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    mat.roughness = 1.0
    mat.metallic = 0.8
    mat.base_color_texture = rgba_texture(rgba_array(shade))
    mat.roughness_texture = rgba_texture(rgba_array(np.repeat(roughness[..., None], 3, axis=-1)))
    return mat


@dataclass
class _ConveyorMaterials:
    belt_side: sapien.render.RenderMaterial
    belt_top: sapien.render.RenderMaterial
    belt_textures: _BeltTextures
    frame: sapien.render.RenderMaterial
    roller: sapien.render.RenderMaterial
    motor: sapien.render.RenderMaterial


_MATERIALS: dict[int, _ConveyorMaterials] = {}


def _get_materials(texture_slot: int = 0) -> _ConveyorMaterials:
    cached = _MATERIALS.get(texture_slot)
    if cached is None:
        rng = np.random.default_rng(_TEXTURE_SEED)
        belt_top, belt_textures = _make_belt_material(rng)
        cached = _ConveyorMaterials(
            belt_side=flat((0.06, 0.06, 0.07, 1.0), roughness=0.9, metallic=0.0),
            belt_top=belt_top,
            belt_textures=belt_textures,
            frame=_make_steel_material(rng),
            roller=flat((0.55, 0.55, 0.58, 1.0), roughness=0.35, metallic=0.85),
            motor=flat((0.18, 0.18, 0.20, 1.0), roughness=0.6, metallic=0.3),
        )
        _MATERIALS[texture_slot] = cached
    return cached


class BeltSurfaceScroller:
    def __init__(
        self,
        shapes: list[sapien.render.RenderShapeTriangleMesh],
        base_v: np.ndarray,
        period_m: float,
        belt_textures: _BeltTextures,
    ):
        self._shapes = shapes
        self._base_v = base_v
        self._period_m = period_m
        self._belt_textures = belt_textures
        self._offsets = np.zeros(len(shapes), dtype=np.float64)
        self._dirs = np.ones(len(shapes), dtype=np.float64)
        self._mode: str | None = None
        self._buffers: list[torch.Tensor] | None = None
        self._base_v_fwd: torch.Tensor | None = None
        self._base_v_rev: torch.Tensor | None = None
        self._last_upload: tuple[int, float] | None = None

    def _resolve_mode(self) -> str:
        if self._mode is None:
            import torch

            try:
                self._buffers = [shape.parts[0].cuda_vertices.torch() for shape in self._shapes]
            except Exception as e:  # noqa: BLE001 — sapien raises plain RuntimeError without CUDA interop
                logger.warning("No CUDA vertex buffer access; belt scrolls by texture re-upload instead: %s", e)
                self._mode = "upload"
                return self._mode
            self._base_v_fwd = torch.as_tensor(
                self._base_v, device=self._buffers[0].device, dtype=self._buffers[0].dtype
            )
            self._base_v_rev = float(self._base_v.max()) - self._base_v_fwd
            self._mode = "cuda"
        return self._mode

    def _write(self, env_indices: np.ndarray | None = None) -> None:
        mode = self._resolve_mode()
        if mode == "cuda":
            self._write_cuda(env_indices)
        elif mode == "upload":
            self._write_upload()

    def _write_cuda(self, env_indices: np.ndarray | None) -> None:
        import torch

        assert self._buffers is not None and self._base_v_fwd is not None and self._base_v_rev is not None
        indices = range(len(self._buffers)) if env_indices is None else env_indices
        for i in indices:
            base = self._base_v_fwd if self._dirs[i] > 0 else self._base_v_rev
            self._buffers[i][:, _VERTEX_V_COL] = base + self._offsets[i]
        torch.cuda.current_stream().synchronize()

    def _write_upload(self) -> None:
        res = self._belt_textures.arrays[0].shape[0]
        phase = int(round(self._offsets[0] * res)) % res
        direction = self._dirs[0]
        if (phase, direction) == self._last_upload:
            return
        self._last_upload = (phase, direction)
        shift = -phase if direction > 0 else phase
        try:
            for tex, arr in zip(self._belt_textures.textures, self._belt_textures.arrays, strict=True):
                img = arr if direction > 0 else np.flip(arr, axis=0)
                tex.upload(np.ascontiguousarray(np.roll(img, shift, axis=0)))
        except Exception as e:  # noqa: BLE001 — surface once, then go static
            logger.warning("Belt texture scroll disabled (texture upload failed): %s", e)
            self._mode = "off"

    def advance(self, surface_dy_m: np.ndarray) -> None:
        self._offsets = (self._offsets - self._dirs * surface_dy_m / self._period_m) % 1.0
        self._write()

    def reset(self, env_idx: np.ndarray, directions: np.ndarray) -> None:
        self._offsets[env_idx] = 0.0
        self._dirs[env_idx] = directions
        self._write(env_idx)


@dataclass
class ConveyorBelt:
    actor: Actor
    scroller: BeltSurfaceScroller
    drive: BeltDrive
    control_dt: float

    @property
    def velocity(self) -> torch.Tensor:
        return self.drive.velocity

    @property
    def direction_sign(self) -> torch.Tensor:
        return self.drive.direction_sign

    @property
    def primary_rider(self) -> Actor:
        return self.drive.primary_rider

    @property
    def riders(self) -> list[Actor]:
        return self.drive.riders

    def register_rider(self, actor: Actor, *, patch_radius: float | torch.Tensor) -> None:
        self.drive.register_rider(actor, patch_radius=patch_radius)

    def reset(self, env_idx: torch.Tensor, velocity: torch.Tensor) -> None:
        self.drive.reset(env_idx, velocity)
        idx_np = env_idx.cpu().numpy()
        self.scroller.reset(idx_np, np.where(self.drive.velocity_np[idx_np] >= 0.0, 1.0, -1.0))

    def on_control_step(self) -> None:
        self.scroller.advance(self.drive.velocity_np * self.control_dt)

    def on_simulation_step(self) -> None:
        self.drive.step()


def step_belts(*belts: ConveyorBelt) -> None:
    per_actor: dict[int, tuple[Actor, torch.Tensor, torch.Tensor]] = {}
    for belt in belts:
        for actor, force, torque in belt.drive.compute_writes():
            prev = per_actor.get(id(actor))
            per_actor[id(actor)] = (
                (actor, force, torque) if prev is None else (actor, prev[1] + force, prev[2] + torque)
            )
    if per_actor:
        apply_drive_writes(belts[0].actor.scene, list(per_actor.values()))


def build_conveyor(
    scene: ManiSkillScene,
    *,
    workspace_offset: tuple[float, float, float],
    deck_height: float,
    belt_half_width: float,
    half_length: float,
    static_friction: float,
    kinetic_friction: float,
    name: str = "conveyor",
    yaw: float = 0.0,
    texture_slot: int = 0,
) -> ConveyorBelt:
    rail_skirt = 0.03
    if half_length <= _ROLLER_RADIUS + 0.10:
        raise ValueError(f"half_length={half_length} too short for the {_ROLLER_RADIUS} end rollers + motor")
    if deck_height <= 2 * _ROLLER_RADIUS + rail_skirt:
        raise ValueError(f"deck_height={deck_height} leaves no room for legs under the belt band")

    ox, oy, oz = workspace_offset

    mats = _get_materials(texture_slot)
    belt_mat = mats.belt_side
    belt_top_mat, belt_textures = mats.belt_top, mats.belt_textures
    frame_mat = mats.frame
    roller_mat = mats.roller
    motor_mat = mats.motor

    r = _ROLLER_RADIUS
    band_center_z = deck_height - r
    band_half_length = half_length - r

    builder = scene.create_actor_builder()

    builder.add_box_collision(
        pose=sapien.Pose(p=[0.0, 0.0, band_center_z]),
        half_size=[belt_half_width, half_length, r],
        material=sapien.physx.PhysxMaterial(static_friction=0.0, dynamic_friction=0.0, restitution=0.0),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, band_center_z]),
        half_size=[belt_half_width, band_half_length, r],
        material=belt_mat,
        name="belt_band",
    )

    for sign in (-1.0, 1.0):
        builder.add_cylinder_visual(
            pose=sapien.Pose(p=[0.0, sign * band_half_length, band_center_z]),
            radius=r,
            half_length=belt_half_width,
            material=roller_mat,
            name=f"roller_{'pos' if sign > 0 else 'neg'}",
        )

    rail_half_height = (2 * r + _RAIL_LIP + rail_skirt) / 2
    rail_top_z = deck_height + _RAIL_LIP
    rail_x = belt_half_width + _RAIL_HALF_THICKNESS
    for sign in (-1.0, 1.0):
        builder.add_box_visual(
            pose=sapien.Pose(p=[sign * rail_x, 0.0, rail_top_z - rail_half_height]),
            half_size=[_RAIL_HALF_THICKNESS, half_length, rail_half_height],
            material=frame_mat,
            name=f"rail_{'pos' if sign > 0 else 'neg'}",
        )

    leg_top_z = rail_top_z - 2 * rail_half_height
    leg_end_y = max(half_length - 2 * _LEG_HALF_XSECTION - 0.04, 0.0)
    n_pairs = max(2, math.ceil(2 * leg_end_y / _LEG_MAX_SPACING) + 1)
    for i in range(n_pairs):
        leg_y = -leg_end_y + i * (2 * leg_end_y / (n_pairs - 1))
        for sx in (-1.0, 1.0):
            builder.add_box_visual(
                pose=sapien.Pose(p=[sx * rail_x, leg_y, leg_top_z / 2]),
                half_size=[_LEG_HALF_XSECTION, _LEG_HALF_XSECTION, leg_top_z / 2],
                material=frame_mat,
                name=f"leg_{i}{'p' if sx > 0 else 'n'}",
            )

    motor_half = (0.035, 0.06, 0.035)
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, half_length - 0.10, band_center_z - r - motor_half[2]]),
        half_size=list(motor_half),
        material=motor_mat,
        name="motor",
    )

    builder.set_initial_pose(sapien.Pose(p=[ox, oy, oz], q=[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]))
    actor = builder.build_kinematic(name=name)

    quad_z = deck_height + 5e-4
    vertices = np.array(
        [
            [-belt_half_width, -band_half_length, quad_z],
            [belt_half_width, -band_half_length, quad_z],
            [belt_half_width, band_half_length, quad_z],
            [-belt_half_width, band_half_length, quad_z],
        ],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (4, 1))
    v_repeats = 2 * band_half_length / _BELT_TEXTURE_PERIOD_M
    base_v = np.array([0.0, 0.0, v_repeats, v_repeats], dtype=np.float32)
    uvs = np.stack([np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32), base_v], axis=1)

    shapes: list[sapien.render.RenderShapeTriangleMesh] = []
    for obj in actor._objs:
        shape = sapien.render.RenderShapeTriangleMesh(
            vertices=vertices, triangles=triangles, normals=normals, uvs=uvs, material=belt_top_mat
        )
        component = sapien.render.RenderBodyComponent()
        component.attach(shape)
        obj.add_component(component)
        shapes.append(shape)

    scroller = BeltSurfaceScroller(shapes, base_v=base_v, period_m=_BELT_TEXTURE_PERIOD_M, belt_textures=belt_textures)
    drive = BeltDrive(
        scene,
        actor,
        deck_top_z=oz + deck_height,
        static_friction=static_friction,
        kinetic_friction=kinetic_friction,
        axis=(-math.sin(yaw), math.cos(yaw)),
    )
    return ConveyorBelt(actor=actor, scroller=scroller, drive=drive, control_dt=1.0 / scene.sim_config.control_freq)
