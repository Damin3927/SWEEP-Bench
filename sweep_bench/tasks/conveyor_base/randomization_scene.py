from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sapien
import sapien.render
from mani_skill.utils.building.actors import build_box

from sweep_bench.tasks.conveyor_base.cameras import EXTERIOR_RIG

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.structs.actor import Actor

_LIGHT_COLOR_SCALE_RANGE = (0.85, 1.15)
_CLUTTER_HALF_SIZE_RANGE = (0.015, 0.035)
_CLUTTER_X_MARGIN = (0.08, 0.24)
_CLUTTER_Y_MARGIN = 0.05


@dataclass
class SceneRandomizationConfig:
    texture: bool = False
    lighting: bool = False
    camera_jitter: bool = False
    clutter: bool = False

    lighting_intensity_range: tuple[float, float] = (0.7, 1.3)
    lighting_direction_jitter: float = 0.4
    camera_jitter_range: float = 0.05
    max_clutter: int = 4


def sample_tint(rng: np.random.RandomState) -> list[float]:
    hue = float(rng.uniform(0.0, 1.0))
    r, g, b = colorsys.hsv_to_rgb(hue, 0.45, 0.85)
    return [r, g, b, 1.0]


def sample_table_material(rng: np.random.RandomState, material_ids: Sequence[str]) -> str:
    index = int(rng.uniform(0.0, 1.0) * len(material_ids))
    return material_ids[min(index, len(material_ids) - 1)]


def build_texture_overlay(
    scene: ManiSkillScene,
    rgba: list[float],
    *,
    workspace_offset: tuple[float, float, float],
    half_size: tuple[float, float],
    center_dx: float,
) -> Actor:
    ox, oy, oz = workspace_offset
    hx, hy = half_size
    overlay_half_z = 0.001
    return build_box(
        scene,
        half_sizes=[hx, hy, overlay_half_z],
        color=rgba,
        name="texture_overlay",
        body_type="kinematic",
        add_collision=False,
        initial_pose=sapien.Pose(p=[ox + center_dx, oy, oz + overlay_half_z]),
    )


def sample_lighting_params(
    rng: np.random.RandomState, cfg: SceneRandomizationConfig
) -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
    intensity_scale = float(rng.uniform(*cfg.lighting_intensity_range))
    c = rng.uniform(*_LIGHT_COLOR_SCALE_RANGE, size=3)
    color_scale = (float(c[0]), float(c[1]), float(c[2]))
    j = cfg.lighting_direction_jitter
    d = rng.uniform(-j, j, size=3)
    key_dir_offset = (float(d[0]), float(d[1]), float(d[2]))
    return intensity_scale, color_scale, key_dir_offset


def sample_camera_jitter(
    rng: np.random.RandomState, cfg: SceneRandomizationConfig
) -> tuple[tuple[float, float, float], ...]:
    j = cfg.camera_jitter_range
    d = rng.uniform(-j, j, size=(len(EXTERIOR_RIG), 3))
    return tuple((float(row[0]), float(row[1]), float(row[2])) for row in d)


def sample_clutter_count(rng: np.random.RandomState, cfg: SceneRandomizationConfig) -> int:
    return int(rng.randint(0, cfg.max_clutter + 1))


def scene_randomization_active(cfg: SceneRandomizationConfig) -> bool:
    return cfg.texture or cfg.lighting or cfg.camera_jitter or cfg.clutter


def build_clutter(
    scene: ManiSkillScene,
    rng: np.random.RandomState,
    n: int,
    *,
    workspace_offset: tuple[float, float, float],
    belt_half_width: float,
    half_length: float,
) -> list[Actor]:
    ox, oy, oz = workspace_offset
    x_lo = belt_half_width + _CLUTTER_X_MARGIN[0]
    x_hi = belt_half_width + _CLUTTER_X_MARGIN[1]
    props: list[Actor] = []
    for i in range(n):
        half = float(rng.uniform(*_CLUTTER_HALF_SIZE_RANGE))
        x = ox + float(rng.uniform(x_lo, x_hi))
        y = oy + float(rng.uniform(-half_length + _CLUTTER_Y_MARGIN, half_length - _CLUTTER_Y_MARGIN))
        z = oz + half
        actor = build_box(
            scene,
            half_sizes=[half, half, half],
            color=sample_tint(rng),
            name=f"clutter_{i}",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[x, y, z]),
        )
        props.append(actor)
    return props
