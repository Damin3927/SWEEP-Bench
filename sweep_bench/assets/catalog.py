from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from mani_skill.utils.building.actors import get_actor_builder

from sweep_bench.assets.peg_hole import build_hole_block, build_peg, hole_block_collision_volume

if TYPE_CHECKING:
    from collections.abc import Callable

    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder


ObjectRole = Literal["graspable_rider", "container", "insert_socket", "tool"]

ObjectSource = Literal["ycb", "procedural", "gso", "robocasa", "polyhaven"]


@dataclass(frozen=True)
class ObjectSpec:
    id: str
    role: ObjectRole
    source: ObjectSource
    builder_fn: Callable[[ManiSkillScene], ActorBuilder]
    bbox: tuple[float, float, float]
    mass: float
    patch_radius: float | None
    grasp_width: float | None
    tags: tuple[str, ...]


_REGISTRY: dict[str, ObjectSpec] = {}


def register_object(spec: ObjectSpec) -> None:
    if spec.id in _REGISTRY:
        raise ValueError(f"object {spec.id!r} already registered")
    _REGISTRY[spec.id] = spec


def get_object(id: str) -> ObjectSpec:  # noqa: A002 — matches `get_task(gym_id)`'s param naming
    if id not in _REGISTRY:
        raise ValueError(f"Unknown object {id!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[id]


def get_objects(role: ObjectRole | None = None) -> tuple[ObjectSpec, ...]:
    specs = tuple(_REGISTRY.values())
    if role is None:
        return specs
    return tuple(spec for spec in specs if spec.role == role)


_FRUIT_PATCH_RADIUS = 0.02

register_object(
    ObjectSpec(
        id="apple",
        role="graspable_rider",
        source="ycb",
        builder_fn=functools.partial(get_actor_builder, id="ycb:013_apple"),
        bbox=(0.0379, 0.0379, 0.0361),
        mass=0.2647,
        patch_radius=_FRUIT_PATCH_RADIUS,
        grasp_width=0.0758,
        tags=("ycb", "fruit", "round"),
    )
)
register_object(
    ObjectSpec(
        id="strawberry",
        role="graspable_rider",
        source="ycb",
        builder_fn=functools.partial(get_actor_builder, id="ycb:012_strawberry"),
        bbox=(0.0238, 0.0218, 0.0236),
        mass=0.0530,
        patch_radius=_FRUIT_PATCH_RADIUS,
        grasp_width=0.0456,
        tags=("ycb", "fruit", "round"),
    )
)
register_object(
    ObjectSpec(
        id="lemon",
        role="graspable_rider",
        source="ycb",
        builder_fn=functools.partial(get_actor_builder, id="ycb:014_lemon"),
        bbox=(0.0304, 0.0302, 0.0267),
        mass=0.1045,
        patch_radius=_FRUIT_PATCH_RADIUS,
        grasp_width=0.0606,
        tags=("ycb", "fruit", "round"),
    )
)
register_object(
    ObjectSpec(
        id="peach",
        role="graspable_rider",
        source="ycb",
        builder_fn=functools.partial(get_actor_builder, id="ycb:015_peach"),
        bbox=(0.0295, 0.0326, 0.0303),
        mass=0.1260,
        patch_radius=_FRUIT_PATCH_RADIUS,
        grasp_width=0.0621,
        tags=("ycb", "fruit", "round"),
    )
)
register_object(
    ObjectSpec(
        id="orange",
        role="graspable_rider",
        source="ycb",
        builder_fn=functools.partial(get_actor_builder, id="ycb:017_orange"),
        bbox=(0.0365, 0.0369, 0.0364),
        mass=0.2168,
        patch_radius=_FRUIT_PATCH_RADIUS,
        grasp_width=0.0734,
        tags=("ycb", "fruit", "round"),
    )
)
register_object(
    ObjectSpec(
        id="plum",
        role="graspable_rider",
        source="ycb",
        builder_fn=functools.partial(get_actor_builder, id="ycb:018_plum"),
        bbox=(0.0284, 0.0270, 0.0267),
        mass=0.0930,
        patch_radius=_FRUIT_PATCH_RADIUS,
        grasp_width=0.0554,
        tags=("ycb", "fruit", "round"),
    )
)

register_object(
    ObjectSpec(
        id="bowl",
        role="container",
        source="ycb",
        builder_fn=functools.partial(get_actor_builder, id="ycb:024_bowl"),
        bbox=(0.0811, 0.0811, 0.0277),
        mass=0.2332,
        patch_radius=0.04,
        grasp_width=None,
        tags=("ycb", "container"),
    )
)

_PEG_RADIUS = 0.0025
_PEG_HALF_LENGTH = 0.04
register_object(
    ObjectSpec(
        id="peg",
        role="tool",
        source="procedural",
        builder_fn=functools.partial(build_peg, radius=_PEG_RADIUS, half_length=_PEG_HALF_LENGTH),
        bbox=(_PEG_RADIUS, _PEG_RADIUS, _PEG_HALF_LENGTH),
        mass=0.00157,
        patch_radius=None,
        grasp_width=2 * (_PEG_RADIUS - 0.0005),
        tags=("procedural", "peg", "cylinder"),
    )
)

_BLOCK_HALF_EXTENT = 0.04
_BLOCK_HALF_HEIGHT = 0.015
_BLOCK_CHAMFER = 0.006
_DEFAULT_HOLE_HALF_WIDTH = _PEG_RADIUS + 0.0055
_BLOCK_DENSITY = 1000.0
register_object(
    ObjectSpec(
        id="hole_block",
        role="insert_socket",
        source="procedural",
        builder_fn=functools.partial(
            build_hole_block,
            hole_half_width=_DEFAULT_HOLE_HALF_WIDTH,
            block_half_extent=_BLOCK_HALF_EXTENT,
            block_half_height=_BLOCK_HALF_HEIGHT,
            chamfer=_BLOCK_CHAMFER,
        ),
        bbox=(_BLOCK_HALF_EXTENT, _BLOCK_HALF_EXTENT, _BLOCK_HALF_HEIGHT),
        mass=_BLOCK_DENSITY
        * hole_block_collision_volume(
            hole_half_width=_DEFAULT_HOLE_HALF_WIDTH,
            block_half_extent=_BLOCK_HALF_EXTENT,
            block_half_height=_BLOCK_HALF_HEIGHT,
            chamfer=_BLOCK_CHAMFER,
        ),
        patch_radius=0.025,
        grasp_width=None,
        tags=("procedural", "socket", "insert"),
    )
)
