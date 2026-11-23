from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from sweep_bench.assets.catalog import ObjectSpec
from sweep_bench.tasks.conveyor_handle_grasp.assets import (
    POST_ROOT_DEPTH,
    build_bail_vessel,
    ring_segment_half_length,
)


@dataclass(frozen=True)
class HandleVessel:
    id: str
    body_radius: float
    body_half_height: float
    wall_thickness: float
    floor_half_thickness: float
    bar_radius: float
    aperture: float
    color: tuple[float, float, float, float]

    @property
    def bar_local_z(self) -> float:
        return self.body_half_height + self.aperture + self.bar_radius

    @property
    def grasp_width(self) -> float:
        return 2.0 * self.bar_radius

    @property
    def top_local_z(self) -> float:
        return self.bar_local_z + self.bar_radius

    @property
    def mass(self) -> float:
        seg_half = ring_segment_half_length(self.body_radius, self.wall_thickness)
        walls = 12.0 * self.wall_thickness * (2.0 * seg_half) * (2.0 * self.body_half_height)
        floor = math.pi * self.body_radius**2 * (2.0 * self.floor_half_thickness)
        post_len = self.bar_local_z - max(-self.body_half_height, self.body_half_height - POST_ROOT_DEPTH)
        posts = 2.0 * math.pi * self.bar_radius**2 * post_len
        bar = math.pi * self.bar_radius**2 * (2.0 * self.body_radius)
        return 1000.0 * (walls + floor + posts + bar)

    def spec(self) -> ObjectSpec:
        half_z = (self.top_local_z + self.body_half_height) / 2.0
        return ObjectSpec(
            id=self.id,
            role="graspable_rider",
            source="procedural",
            builder_fn=functools.partial(
                build_bail_vessel,
                body_radius=self.body_radius,
                body_half_height=self.body_half_height,
                wall_thickness=self.wall_thickness,
                floor_half_thickness=self.floor_half_thickness,
                bar_radius=self.bar_radius,
                aperture=self.aperture,
                color=self.color,
            ),
            bbox=(self.body_radius, self.body_radius, half_z),
            mass=self.mass,
            patch_radius=self.body_radius,
            grasp_width=self.grasp_width,
            tags=("procedural", "bail_vessel", self.id),
        )


_STEEL = (0.62, 0.65, 0.70, 1.0)
_GREEN = (0.16, 0.46, 0.30, 1.0)
_RED = (0.72, 0.22, 0.18, 1.0)
_BLUE = (0.20, 0.34, 0.62, 1.0)

HANDLE_VESSELS: tuple[HandleVessel, ...] = (
    HandleVessel(
        id="pail",
        body_radius=0.038,
        body_half_height=0.018,
        wall_thickness=0.006,
        floor_half_thickness=0.003,
        bar_radius=0.008,
        aperture=0.050,
        color=_STEEL,
    ),
    HandleVessel(
        id="bucket",
        body_radius=0.044,
        body_half_height=0.020,
        wall_thickness=0.006,
        floor_half_thickness=0.003,
        bar_radius=0.009,
        aperture=0.050,
        color=_BLUE,
    ),
    HandleVessel(
        id="kettle",
        body_radius=0.032,
        body_half_height=0.020,
        wall_thickness=0.005,
        floor_half_thickness=0.003,
        bar_radius=0.006,
        aperture=0.050,
        color=_GREEN,
    ),
    HandleVessel(
        id="basket",
        body_radius=0.041,
        body_half_height=0.015,
        wall_thickness=0.005,
        floor_half_thickness=0.003,
        bar_radius=0.007,
        aperture=0.052,
        color=_RED,
    ),
)

HANDLE_VESSEL_BY_ID: dict[str, HandleVessel] = {v.id: v for v in HANDLE_VESSELS}

HANDLE_VESSEL_SPECS: tuple[ObjectSpec, ...] = tuple(v.spec() for v in HANDLE_VESSELS)

HANDLE_GRASP_OBJECT_IDS: tuple[str, ...] = tuple(v.id for v in HANDLE_VESSELS)
