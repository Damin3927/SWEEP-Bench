from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import sapien
import sapien.render

from sweep_bench.assets.catalog import ObjectSpec, register_object
from sweep_bench.assets.materials import Finish, flat, textured
from sweep_bench.assets.peg_hole import build_hole_block, hole_block_collision_volume

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.actor_builder import ActorBuilder


_DENSITY = 1000.0

_BELT_HALF_WIDTH = 0.08
_MAX_RIDER_FOOTPRINT = _BELT_HALF_WIDTH - 0.01

_MAX_GRASP_WIDTH = 0.07

_UPRIGHT = sapien.Pose(q=[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])

_COLORS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.12, 0.12, 1.0),
    "green": (0.13, 0.60, 0.20, 1.0),
    "blue": (0.13, 0.30, 0.75, 1.0),
    "yellow": (0.90, 0.80, 0.10, 1.0),
    "orange": (0.92, 0.50, 0.12, 1.0),
    "purple": (0.50, 0.20, 0.60, 1.0),
    "cyan": (0.10, 0.65, 0.70, 1.0),
    "gray": (0.50, 0.50, 0.52, 1.0),
}
_COLOR_NAMES = tuple(_COLORS)


def _surface(
    color: tuple[float, float, float, float], finish: Finish | None, finish_key: str
) -> sapien.render.RenderMaterial:
    return flat(color) if finish is None else textured(color, finish, key=finish_key)


def _mm(x: float) -> str:
    return f"{round(x * 1000):03d}"


def build_box(
    scene: ManiSkillScene,
    *,
    half_x: float,
    half_y: float,
    half_z: float,
    color: tuple[float, float, float, float],
    finish: Finish | None = None,
    finish_key: str = "",
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    half = [half_x, half_y, half_z]
    builder.add_box_collision(half_size=half)
    builder.add_box_visual(half_size=half, material=_surface(color, finish, finish_key))
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_can(
    scene: ManiSkillScene,
    *,
    radius: float,
    half_length: float,
    color: tuple[float, float, float, float],
    finish: Finish | None = None,
    finish_key: str = "",
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    builder = scene.create_actor_builder()
    builder.add_cylinder_collision(pose=_UPRIGHT, radius=radius, half_length=half_length)
    builder.add_cylinder_visual(
        pose=_UPRIGHT, radius=radius, half_length=half_length, material=_surface(color, finish, finish_key)
    )
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_bin(
    scene: ManiSkillScene,
    *,
    inner_half_x: float,
    inner_half_y: float,
    wall_thickness: float,
    wall_half_height: float,
    floor_half_height: float,
    color: tuple[float, float, float, float],
    finish: Finish | None = None,
    finish_key: str = "",
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    wt = wall_thickness
    outer_half_x = inner_half_x + 2 * wt
    outer_half_y = inner_half_y + 2 * wt
    mat = _surface(color, finish, finish_key)
    builder = scene.create_actor_builder()

    floor = [outer_half_x, outer_half_y, floor_half_height]
    builder.add_box_collision(half_size=floor)
    builder.add_box_visual(half_size=floor, material=mat)

    wall_z = floor_half_height + wall_half_height
    walls = [
        ((wt, outer_half_y, wall_half_height), [inner_half_x + wt, 0.0, wall_z]),
        ((wt, outer_half_y, wall_half_height), [-(inner_half_x + wt), 0.0, wall_z]),
        ((inner_half_x, wt, wall_half_height), [0.0, inner_half_y + wt, wall_z]),
        ((inner_half_x, wt, wall_half_height), [0.0, -(inner_half_y + wt), wall_z]),
    ]
    for half, p in walls:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=list(half))
        builder.add_box_visual(pose=pose, half_size=list(half), material=mat)

    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_plate(
    scene: ManiSkillScene,
    *,
    radius: float,
    half_thickness: float,
    color: tuple[float, float, float, float],
    finish: Finish | None = None,
    finish_key: str = "",
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    return build_can(
        scene,
        radius=radius,
        half_length=half_thickness,
        color=color,
        finish=finish,
        finish_key=finish_key,
        initial_pose=initial_pose,
    )


def build_prism_peg(
    scene: ManiSkillScene,
    *,
    half_extent: float,
    base_half_width: float,
    half_length: float,
    color: tuple[float, float, float, float],
    n_slabs: int = 4,
    finish: Finish | None = None,
    finish_key: str = "",
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    mat = _surface(color, finish, finish_key)
    builder = scene.create_actor_builder()
    slab_half_x = half_extent / n_slabs
    for hs, p in _prism_slabs(half_extent, base_half_width, half_length, n_slabs, slab_half_x):
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=list(hs))
        builder.add_box_visual(pose=pose, half_size=list(hs), material=mat)
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def _prism_slabs(
    half_extent: float, base_half_width: float, half_length: float, n_slabs: int, slab_half_x: float
) -> Iterator[tuple[tuple[float, float, float], list[float]]]:
    for i in range(n_slabs):
        x_c = -half_extent + (2 * i + 1) * slab_half_x
        y_half = max(base_half_width * (half_extent - x_c) / (2 * half_extent), 1e-4)
        yield (slab_half_x, y_half, half_length), [x_c, 0.0, 0.0]


def build_star_peg(
    scene: ManiSkillScene,
    *,
    core_half: float,
    arm_half_length: float,
    arm_half_width: float,
    half_length: float,
    color: tuple[float, float, float, float],
    finish: Finish | None = None,
    finish_key: str = "",
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    mat = _surface(color, finish, finish_key)
    builder = scene.create_actor_builder()
    arm_c = core_half + arm_half_length
    prims = [
        ((core_half, core_half, half_length), [0.0, 0.0, 0.0]),
        ((arm_half_length, arm_half_width, half_length), [arm_c, 0.0, 0.0]),
        ((arm_half_length, arm_half_width, half_length), [-arm_c, 0.0, 0.0]),
        ((arm_half_width, arm_half_length, half_length), [0.0, arm_c, 0.0]),
        ((arm_half_width, arm_half_length, half_length), [0.0, -arm_c, 0.0]),
    ]
    for hs, p in prims:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=list(hs))
        builder.add_box_visual(pose=pose, half_size=list(hs), material=mat)
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_moon_peg(
    scene: ManiSkillScene,
    *,
    arc_radius: float,
    radial_half: float,
    half_length: float,
    color: tuple[float, float, float, float],
    n_segments: int = 5,
    arc_span: float = math.radians(75.0),
    finish: Finish | None = None,
    finish_key: str = "",
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    mat = _surface(color, finish, finish_key)
    builder = scene.create_actor_builder()
    for hs, p, q in _moon_segments(arc_radius, radial_half, half_length, n_segments, arc_span):
        pose = sapien.Pose(p=p, q=q)
        builder.add_box_collision(pose=pose, half_size=list(hs))
        builder.add_box_visual(pose=pose, half_size=list(hs), material=mat)
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def _moon_segments(
    arc_radius: float, radial_half: float, half_length: float, n_segments: int, arc_span: float
) -> Iterator[tuple[tuple[float, float, float], list[float], list[float]]]:
    d_theta = 2 * arc_span / (n_segments - 1) if n_segments > 1 else 0.0
    tangential_half = 0.9 * arc_radius * d_theta / 2 if n_segments > 1 else arc_radius
    for i in range(n_segments):
        theta = -arc_span + i * d_theta
        p = [arc_radius * math.cos(theta), arc_radius * math.sin(theta), 0.0]
        q = [math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)]
        yield (radial_half, tangential_half, half_length), p, q


def build_lid(
    scene: ManiSkillScene,
    *,
    radius: float,
    cap_half_thickness: float,
    plug_radius: float,
    plug_half_length: float,
    color: tuple[float, float, float, float],
    finish: Finish | None = None,
    finish_key: str = "",
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    mat = _surface(color, finish, finish_key)
    builder = scene.create_actor_builder()
    builder.add_cylinder_collision(pose=_UPRIGHT, radius=radius, half_length=cap_half_thickness)
    builder.add_cylinder_visual(pose=_UPRIGHT, radius=radius, half_length=cap_half_thickness, material=mat)
    plug_pose = sapien.Pose(p=[0.0, 0.0, -(cap_half_thickness + plug_half_length)], q=_UPRIGHT.q)
    builder.add_cylinder_collision(pose=plug_pose, radius=plug_radius, half_length=plug_half_length)
    builder.add_cylinder_visual(pose=plug_pose, radius=plug_radius, half_length=plug_half_length, material=mat)
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def build_hook(
    scene: ManiSkillScene,
    *,
    shaft_half_length: float,
    rod_half: float,
    arm_half_length: float,
    color: tuple[float, float, float, float],
    finish: Finish | None = None,
    finish_key: str = "",
    initial_pose: sapien.Pose | None = None,
) -> ActorBuilder:
    mat = _surface(color, finish, finish_key)
    builder = scene.create_actor_builder()
    prims = [
        ((shaft_half_length, rod_half, rod_half), [0.0, 0.0, 0.0]),
        ((rod_half, arm_half_length, rod_half), [shaft_half_length - rod_half, rod_half + arm_half_length, 0.0]),
    ]
    for hs, p in prims:
        pose = sapien.Pose(p=p)
        builder.add_box_collision(pose=pose, half_size=list(hs))
        builder.add_box_visual(pose=pose, half_size=list(hs), material=mat)
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder


def _box_volume(half_x: float, half_y: float, half_z: float) -> float:
    return 8 * half_x * half_y * half_z


def _cylinder_volume(radius: float, half_length: float) -> float:
    return math.pi * radius**2 * (2 * half_length)


def _bin_volume(
    inner_half_x: float, inner_half_y: float, wall_thickness: float, wall_half_height: float, floor_half_height: float
) -> float:
    wt = wall_thickness
    outer_half_x = inner_half_x + 2 * wt
    outer_half_y = inner_half_y + 2 * wt
    floor = _box_volume(outer_half_x, outer_half_y, floor_half_height)
    x_walls = 2 * _box_volume(wt, outer_half_y, wall_half_height)
    y_walls = 2 * _box_volume(inner_half_x, wt, wall_half_height)
    return floor + x_walls + y_walls


def _socket_volume(block_half_extent: float, hole_half_width: float, block_half_height: float) -> float:
    return hole_block_collision_volume(
        hole_half_width=hole_half_width,
        block_half_extent=block_half_extent,
        block_half_height=block_half_height,
    )


def _prism_volume(half_extent: float, base_half_width: float, half_length: float, n_slabs: int) -> float:
    slab_half_x = half_extent / n_slabs
    return sum(
        _box_volume(*hs) for hs, _ in _prism_slabs(half_extent, base_half_width, half_length, n_slabs, slab_half_x)
    )


def _star_volume(core_half: float, arm_half_length: float, arm_half_width: float, half_length: float) -> float:
    core = _box_volume(core_half, core_half, half_length)
    arms = 4 * _box_volume(arm_half_length, arm_half_width, half_length)
    return core + arms


def _moon_volume(arc_radius: float, radial_half: float, half_length: float, n_segments: int, arc_span: float) -> float:
    return sum(
        _box_volume(*hs) for hs, _, _ in _moon_segments(arc_radius, radial_half, half_length, n_segments, arc_span)
    )


def _lid_volume(radius: float, cap_half_thickness: float, plug_radius: float, plug_half_length: float) -> float:
    return _cylinder_volume(radius, cap_half_thickness) + _cylinder_volume(plug_radius, plug_half_length)


def _hook_volume(shaft_half_length: float, rod_half: float, arm_half_length: float) -> float:
    shaft = _box_volume(shaft_half_length, rod_half, rod_half)
    arm = _box_volume(rod_half, arm_half_length, rod_half)
    return shaft + arm


_CUBE_FULL = (0.03, 0.04, 0.05, 0.06)
_CUBE_COLORS = _COLOR_NAMES[:4]
_RECT_BOXES = ((0.05, 0.03, 0.025), (0.06, 0.04, 0.03), (0.07, 0.04, 0.02), (0.05, 0.05, 0.02))
_RECT_COLORS = _COLOR_NAMES[:3]

_CANS = ((0.02, 0.015), (0.025, 0.02), (0.03, 0.02), (0.03, 0.025))
_CAN_COLORS = _COLOR_NAMES[:4]

_BIN_WALL_THICKNESS = 0.005
_BIN_WALL_HALF_HEIGHT = 0.015
_BIN_FLOOR_HALF_HEIGHT = 0.005
_BINS = ((0.03, 0.03), (0.04, 0.03), (0.05, 0.04), (0.055, 0.045))
_BIN_COLORS = _COLOR_NAMES[:3]

_PLATES = ((0.04, 0.004), (0.05, 0.005), (0.06, 0.005), (0.065, 0.006))
_PLATE_COLORS = _COLOR_NAMES[:3]

_SOCKET_BLOCK_HALF_EXTENT = 0.04
_SOCKET_BLOCK_HALF_HEIGHT = 0.015
_SOCKET_TOLERANCES = (0.004, 0.008)
_PEG_COLORS = _COLOR_NAMES[:3]
_PEG_HALF_LENGTH = 0.02


@dataclass(frozen=True)
class _PegShape:
    build: Callable[..., ActorBuilder]
    bbox: tuple[float, float, float]
    grasp_half: float
    volume: float
    socket_half: float


def _peg_shapes() -> dict[str, _PegShape]:
    cube_half = 0.012
    cyl_radius = 0.012
    prism_half, prism_base = 0.014, 0.014
    star_core, star_arm_len, star_arm_w = 0.006, 0.0045, 0.004
    star_reach = star_core + 2 * star_arm_len
    moon_rc, moon_radial = 0.010, 0.004
    moon_span = math.radians(75.0)
    moon_seg = 5
    moon_tan = 0.9 * moon_rc * (2 * moon_span / (moon_seg - 1)) / 2
    moon_reach = moon_rc + max(moon_radial, moon_tan)
    return {
        "cube": _PegShape(
            build=functools.partial(build_box, half_x=cube_half, half_y=cube_half, half_z=cube_half),
            bbox=(cube_half, cube_half, cube_half),
            grasp_half=cube_half,
            volume=_box_volume(cube_half, cube_half, cube_half),
            socket_half=cube_half,
        ),
        "cylinder": _PegShape(
            build=functools.partial(build_can, radius=cyl_radius, half_length=_PEG_HALF_LENGTH),
            bbox=(cyl_radius, cyl_radius, _PEG_HALF_LENGTH),
            grasp_half=cyl_radius,
            volume=_cylinder_volume(cyl_radius, _PEG_HALF_LENGTH),
            socket_half=cyl_radius,
        ),
        "prism": _PegShape(
            build=functools.partial(
                build_prism_peg, half_extent=prism_half, base_half_width=prism_base, half_length=_PEG_HALF_LENGTH
            ),
            bbox=(prism_half, prism_base, _PEG_HALF_LENGTH),
            grasp_half=min(prism_half, prism_base),
            volume=_prism_volume(prism_half, prism_base, _PEG_HALF_LENGTH, 4),
            socket_half=max(prism_half, prism_base),
        ),
        "star": _PegShape(
            build=functools.partial(
                build_star_peg,
                core_half=star_core,
                arm_half_length=star_arm_len,
                arm_half_width=star_arm_w,
                half_length=_PEG_HALF_LENGTH,
            ),
            bbox=(star_reach, star_reach, _PEG_HALF_LENGTH),
            grasp_half=star_reach,
            volume=_star_volume(star_core, star_arm_len, star_arm_w, _PEG_HALF_LENGTH),
            socket_half=star_reach,
        ),
        "moon": _PegShape(
            build=functools.partial(
                build_moon_peg, arc_radius=moon_rc, radial_half=moon_radial, half_length=_PEG_HALF_LENGTH
            ),
            bbox=(moon_reach, moon_reach, _PEG_HALF_LENGTH),
            grasp_half=moon_reach,
            volume=_moon_volume(moon_rc, moon_radial, _PEG_HALF_LENGTH, moon_seg, moon_span),
            socket_half=moon_reach,
        ),
    }


_LIDS = ((0.025, 0.004, 0.018, 0.006), (0.030, 0.005, 0.022, 0.007), (0.035, 0.005, 0.026, 0.008))
_LID_COLORS = _COLOR_NAMES[:3]

_HOOKS = ((0.030, 0.004, 0.015), (0.035, 0.005, 0.020), (0.025, 0.004, 0.012))
_HOOK_COLORS = _COLOR_NAMES[:3]

_FINISH_BY_ROLE_TAG: dict[str, Finish] = {
    "box": "cardboard",
    "bin": "cardboard",
    "can": "plastic",
    "plate": "plastic",
    "lid": "plastic",
    "peg": "plastic",
    "socket": "metal",
    "hook": "metal",
}


def _with_finish(spec: ObjectSpec) -> ObjectSpec:
    finish = _FINISH_BY_ROLE_TAG[spec.tags[1]]
    return replace(spec, builder_fn=functools.partial(spec.builder_fn, finish=finish, finish_key=spec.id))


def _generate_specs() -> tuple[ObjectSpec, ...]:
    specs: list[ObjectSpec] = []

    for full in _CUBE_FULL:
        h = full / 2
        for cname in _CUBE_COLORS:
            specs.append(
                ObjectSpec(
                    id=f"proc_box_{_mm(full)}x{_mm(full)}x{_mm(full)}_{cname}",
                    role="graspable_rider",
                    source="procedural",
                    builder_fn=functools.partial(build_box, half_x=h, half_y=h, half_z=h, color=_COLORS[cname]),
                    bbox=(h, h, h),
                    mass=_DENSITY * _box_volume(h, h, h),
                    patch_radius=h,
                    grasp_width=2 * h,
                    tags=("procedural", "box", "cube", cname),
                )
            )
    for fx, fy, fz in _RECT_BOXES:
        hx, hy, hz = fx / 2, fy / 2, fz / 2
        for cname in _RECT_COLORS:
            specs.append(
                ObjectSpec(
                    id=f"proc_box_{_mm(fx)}x{_mm(fy)}x{_mm(fz)}_{cname}",
                    role="graspable_rider",
                    source="procedural",
                    builder_fn=functools.partial(build_box, half_x=hx, half_y=hy, half_z=hz, color=_COLORS[cname]),
                    bbox=(hx, hy, hz),
                    mass=_DENSITY * _box_volume(hx, hy, hz),
                    patch_radius=min(hx, hy),
                    grasp_width=2 * min(hx, hy),
                    tags=("procedural", "box", cname),
                )
            )
    for radius, half_length in _CANS:
        for cname in _CAN_COLORS:
            specs.append(
                ObjectSpec(
                    id=f"proc_can_{_mm(2 * radius)}x{_mm(2 * half_length)}_{cname}",
                    role="graspable_rider",
                    source="procedural",
                    builder_fn=functools.partial(
                        build_can, radius=radius, half_length=half_length, color=_COLORS[cname]
                    ),
                    bbox=(radius, radius, half_length),
                    mass=_DENSITY * _cylinder_volume(radius, half_length),
                    patch_radius=radius,
                    grasp_width=2 * radius,
                    tags=("procedural", "can", "cylinder", cname),
                )
            )

    for inner_hx, inner_hy in _BINS:
        outer_hx = inner_hx + 2 * _BIN_WALL_THICKNESS
        outer_hy = inner_hy + 2 * _BIN_WALL_THICKNESS
        half_z = _BIN_FLOOR_HALF_HEIGHT + _BIN_WALL_HALF_HEIGHT
        for cname in _BIN_COLORS:
            specs.append(
                ObjectSpec(
                    id=f"proc_bin_{_mm(2 * outer_hx)}x{_mm(2 * outer_hy)}x{_mm(2 * half_z)}_{cname}",
                    role="container",
                    source="procedural",
                    builder_fn=functools.partial(
                        build_bin,
                        inner_half_x=inner_hx,
                        inner_half_y=inner_hy,
                        wall_thickness=_BIN_WALL_THICKNESS,
                        wall_half_height=_BIN_WALL_HALF_HEIGHT,
                        floor_half_height=_BIN_FLOOR_HALF_HEIGHT,
                        color=_COLORS[cname],
                    ),
                    bbox=(outer_hx, outer_hy, half_z),
                    mass=_DENSITY
                    * _bin_volume(
                        inner_hx, inner_hy, _BIN_WALL_THICKNESS, _BIN_WALL_HALF_HEIGHT, _BIN_FLOOR_HALF_HEIGHT
                    ),
                    patch_radius=min(outer_hx, outer_hy),
                    grasp_width=None,
                    tags=("procedural", "bin", "tray", "container", cname),
                )
            )
    for radius, half_thickness in _PLATES:
        for cname in _PLATE_COLORS:
            specs.append(
                ObjectSpec(
                    id=f"proc_plate_{_mm(2 * radius)}x{_mm(2 * half_thickness)}_{cname}",
                    role="container",
                    source="procedural",
                    builder_fn=functools.partial(
                        build_plate, radius=radius, half_thickness=half_thickness, color=_COLORS[cname]
                    ),
                    bbox=(radius, radius, half_thickness),
                    mass=_DENSITY * _cylinder_volume(radius, half_thickness),
                    patch_radius=radius,
                    grasp_width=None,
                    tags=("procedural", "plate", "coaster", "container", cname),
                )
            )

    shapes = _peg_shapes()
    for shape, info in shapes.items():
        for tol in _SOCKET_TOLERANCES:
            hole = info.socket_half + tol
            specs.append(
                ObjectSpec(
                    id=f"proc_socket_{shape}_tol{_mm(tol)}",
                    role="insert_socket",
                    source="procedural",
                    builder_fn=functools.partial(
                        build_hole_block,
                        hole_half_width=hole,
                        block_half_extent=_SOCKET_BLOCK_HALF_EXTENT,
                        block_half_height=_SOCKET_BLOCK_HALF_HEIGHT,
                    ),
                    bbox=(_SOCKET_BLOCK_HALF_EXTENT, _SOCKET_BLOCK_HALF_EXTENT, _SOCKET_BLOCK_HALF_HEIGHT),
                    mass=_DENSITY * _socket_volume(_SOCKET_BLOCK_HALF_EXTENT, hole, _SOCKET_BLOCK_HALF_HEIGHT),
                    patch_radius=0.5 * (_SOCKET_BLOCK_HALF_EXTENT + hole),
                    grasp_width=None,
                    tags=("procedural", "socket", "insert", shape),
                )
            )
    for shape, info in shapes.items():
        for cname in _PEG_COLORS:
            specs.append(
                ObjectSpec(
                    id=f"proc_peg_{shape}_{cname}",
                    role="tool",
                    source="procedural",
                    builder_fn=functools.partial(info.build, color=_COLORS[cname]),
                    bbox=info.bbox,
                    mass=_DENSITY * info.volume,
                    patch_radius=None,
                    grasp_width=2 * info.grasp_half,
                    tags=("procedural", "peg", shape, cname),
                )
            )

    for radius, cap_t, plug_r, plug_hl in _LIDS:
        for cname in _LID_COLORS:
            specs.append(
                ObjectSpec(
                    id=f"proc_lid_{_mm(2 * radius)}_{cname}",
                    role="tool",
                    source="procedural",
                    builder_fn=functools.partial(
                        build_lid,
                        radius=radius,
                        cap_half_thickness=cap_t,
                        plug_radius=plug_r,
                        plug_half_length=plug_hl,
                        color=_COLORS[cname],
                    ),
                    bbox=(radius, radius, cap_t + plug_hl),
                    mass=_DENSITY * _lid_volume(radius, cap_t, plug_r, plug_hl),
                    patch_radius=None,
                    grasp_width=2 * radius,
                    tags=("procedural", "lid", "cap", cname),
                )
            )
    for shaft_hl, rod_half, arm_hl in _HOOKS:
        for cname in _HOOK_COLORS:
            specs.append(
                ObjectSpec(
                    id=f"proc_hook_{_mm(2 * shaft_hl)}_{cname}",
                    role="tool",
                    source="procedural",
                    builder_fn=functools.partial(
                        build_hook,
                        shaft_half_length=shaft_hl,
                        rod_half=rod_half,
                        arm_half_length=arm_hl,
                        color=_COLORS[cname],
                    ),
                    bbox=(shaft_hl, rod_half + arm_hl, rod_half),
                    mass=_DENSITY * _hook_volume(shaft_hl, rod_half, arm_hl),
                    patch_radius=None,
                    grasp_width=2 * rod_half,
                    tags=("procedural", "hook", "rod", cname),
                )
            )

    return tuple(_with_finish(spec) for spec in specs)


PROCEDURAL_SPECS: tuple[ObjectSpec, ...] = _generate_specs()


def register_procedural_objects(register: Callable[[ObjectSpec], None] = register_object) -> None:
    for spec in PROCEDURAL_SPECS:
        register(spec)
