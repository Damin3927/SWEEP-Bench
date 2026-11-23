from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import sapien.render

if TYPE_CHECKING:
    from collections.abc import Iterable

Rgba = tuple[float, float, float, float]

Finish = Literal["cardboard", "plastic", "metal"]

_TEXTURE_RES = 128

_AMPLITUDE: dict[str, float] = {"cardboard": 0.65, "plastic": 0.45, "metal": 0.55}

_NORMAL_STRENGTH: dict[str, float] = {"cardboard": 2.2, "plastic": 1.0, "metal": 1.6}

_ROUGHNESS_SWING = 0.30

_CARDBOARD_FLUTES = 3
_PLASTIC_CYCLES = 2
_METAL_CYCLES = 4

_FINE_WEIGHT = 0.12


def flat(color: Rgba, *, roughness: float = 0.5, metallic: float = 0.0) -> sapien.render.RenderMaterial:
    return sapien.render.RenderMaterial(base_color=list(color), roughness=roughness, metallic=metallic)


def textured(
    color: Rgba, kind: Finish, *, key: str, roughness: float = 0.5, metallic: float = 0.0
) -> sapien.render.RenderMaterial:
    return _textured_cached(color, kind, key, roughness, metallic)


@functools.cache
def _textured_cached(
    color: Rgba, kind: Finish, key: str, roughness: float, metallic: float
) -> sapien.render.RenderMaterial:
    pattern = luminance_pattern(kind, key)
    mat = sapien.render.RenderMaterial()
    mat.base_color = [1.0, 1.0, 1.0, color[3]]
    mat.roughness = 1.0
    mat.metallic = metallic
    mat.base_color_texture = rgba_texture(rgba_array(baked_rgb(color, kind, key)))
    mat.roughness_texture = rgba_texture(
        rgba_array(np.clip(roughness * pattern.roughness, 0.0, 1.0)[..., None].repeat(3, axis=-1))
    )
    mat.normal_texture = rgba_texture(normal_array(pattern.height, strength=_NORMAL_STRENGTH[kind]))
    return mat


@dataclass(frozen=True)
class Pattern:
    shade: np.ndarray
    roughness: np.ndarray
    height: np.ndarray


@functools.cache
def luminance_pattern(kind: Finish, key: str) -> Pattern:
    rng = np.random.default_rng(_seed(kind, key))
    if kind == "cardboard":
        depth = _cardboard_depth(rng)
    elif kind == "plastic":
        depth = _plastic_depth(rng)
    elif kind == "metal":
        depth = _metal_depth(rng)
    else:  # pragma: no cover - Finish is a closed Literal; guard a bad string
        raise ValueError(f"unknown finish {kind!r}; expected one of {sorted(_AMPLITUDE)}")

    shade = 1.0 - _AMPLITUDE[kind] * depth
    roughness = 1.0 + _ROUGHNESS_SWING * (depth - float(depth.mean()))
    return Pattern(shade=shade, roughness=roughness, height=-depth)


def _seed(kind: Finish, key: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{kind}:{key}".encode(), digest_size=8).digest(), "big")


def _unit(a: np.ndarray) -> np.ndarray:
    lo, hi = (float(x) for x in np.percentile(a, [1.0, 99.0]))
    return np.zeros_like(a) if hi - lo < 1e-12 else np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _stretch(a: np.ndarray, axis: int, radius: int) -> np.ndarray:
    return np.stack([np.roll(a, d, axis=axis) for d in range(-radius, radius + 1)]).mean(axis=0)


def _bands(rng: np.random.Generator, cycles: int) -> np.ndarray:
    x = np.arange(_TEXTURE_RES) / _TEXTURE_RES
    profile = _unit(np.sin(2 * np.pi * cycles * x + float(rng.random()) * 2 * np.pi))
    across_u = bool(rng.integers(0, 2))
    return np.broadcast_to(profile[None, :] if across_u else profile[:, None], (_TEXTURE_RES, _TEXTURE_RES))


def _detailed(bands: np.ndarray, fine: np.ndarray) -> np.ndarray:
    return np.clip((1.0 - _FINE_WEIGHT) * bands + _FINE_WEIGHT * fine, 0.0, 1.0)


def _cardboard_depth(rng: np.random.Generator) -> np.ndarray:
    res = _TEXTURE_RES
    fibre = _unit(_stretch(smooth_wrap(rng.random((res, res)), passes=1), axis=1, radius=3))
    return _detailed(_bands(rng, _CARDBOARD_FLUTES), fibre)


def _plastic_depth(rng: np.random.Generator) -> np.ndarray:
    res = _TEXTURE_RES
    speckle = _unit(smooth_wrap(rng.random((res, res)), passes=1))
    return _detailed(_bands(rng, _PLASTIC_CYCLES), speckle)


def _metal_depth(rng: np.random.Generator) -> np.ndarray:
    res = _TEXTURE_RES
    streaks = _unit(smooth_wrap(_stretch(rng.random((res, res)), axis=0, radius=6), passes=1))
    return _detailed(_bands(rng, _METAL_CYCLES), streaks)


def smooth_wrap(a: np.ndarray, passes: int) -> np.ndarray:
    for _ in range(passes):
        acc = np.zeros_like(a)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += np.roll(np.roll(a, dy, axis=0), dx, axis=1)
        a = acc / 9.0
    return a


def rgba_array(rgb: np.ndarray) -> np.ndarray:
    rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = 255
    return rgba


def rgba_texture(rgba: np.ndarray, srgb: bool = False) -> sapien.render.RenderTexture2D:
    return sapien.render.RenderTexture2D(rgba, format="R8G8B8A8Unorm", address_mode="repeat", srgb=srgb)


def normal_array(height: np.ndarray, strength: float) -> np.ndarray:
    dh_du = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) / 2.0
    dh_dv = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) / 2.0
    n = np.stack([-strength * dh_du, -strength * dh_dv, np.ones_like(height)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return rgba_array(n * 0.5 + 0.5)


def baked_rgb(color: Rgba, kind: Finish, key: str) -> np.ndarray:
    return np.asarray(color[:3])[None, None, :] * luminance_pattern(kind, key).shade[..., None]


def hue_saturation(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hi = rgb.max(axis=-1)
    lo = rgb.min(axis=-1)
    span = hi - lo
    safe = np.where(span < 1e-12, 1.0, span)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.select(
        [hi == r, hi == g],
        [(g - b) / safe % 6.0, (b - r) / safe + 2.0],
        default=(r - g) / safe + 4.0,
    )
    hue = np.where(span < 1e-12, 0.0, hue * 60.0)
    return hue, np.where(hi < 1e-12, 0.0, span / np.where(hi < 1e-12, 1.0, hi))


def hue_delta(color: Rgba, kind: Finish, key: str) -> tuple[float, float]:
    baked_hue, baked_sat = hue_saturation(rgba_array(baked_rgb(color, kind, key))[..., :3] / 255.0)
    flat_hue, flat_sat = hue_saturation(np.asarray(color[:3]))
    dh = np.abs((baked_hue - flat_hue + 180.0) % 360.0 - 180.0)
    return float(dh.mean()), float(np.abs(baked_sat - flat_sat).mean())


def finishes() -> Iterable[Finish]:
    return ("cardboard", "plastic", "metal")
