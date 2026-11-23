from __future__ import annotations

import contextlib
import math
import platform
from typing import TYPE_CHECKING, ClassVar, Protocol, cast, overload

import numpy as np
import sapien
import sapien.render
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG
from mani_skill.utils.building.actors import build_box
from mani_skill.utils.geometry.rotation_conversions import quaternion_apply
from mani_skill.utils.scene_builder.ai2thor.variants import ArchitecTHORSceneBuilder
from mani_skill.utils.scene_builder.replicacad.scene_builder import (
    ReplicaCADSceneBuilder,
)
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from sweep_bench.assets.conveyor import ConveyorBelt, build_conveyor
from sweep_bench.assets.table_materials import load_manifest as load_table_materials
from sweep_bench.assets.table_materials import render_material as table_render_material
from sweep_bench.assets.table_platform import build_table_platform, build_textured_overlay
from sweep_bench.assets.vetted import require_servable_catalogs
from sweep_bench.tasks.conveyor_base.cameras import (
    EXTERIOR_RIG,
    build_lighting,
    human_render_camera_config,
    sensor_camera_configs,
    viewer_camera_config,
)
from sweep_bench.tasks.conveyor_base.config import (
    ConveyorMotionRandomization,
    PlatformStyle,
    SceneStyle,
    VelocityRandomizationConfig,
)
from sweep_bench.tasks.conveyor_base.contact_record import CONTACT_FORCE, ContactRecord
from sweep_bench.tasks.conveyor_base.early_fail import (
    HULL_TOLERANCE,
    RETURN_SPEED,
    ROBOT_REACH,
    hull_points,
)
from sweep_bench.tasks.conveyor_base.episode_rng import PerEnvEpisodeRNG
from sweep_bench.tasks.conveyor_base.on_belt import DEFAULT_MAX_RISE, belt_carries, rider_at_rest_height
from sweep_bench.tasks.conveyor_base.placements import (
    Background,
    validate_background_pool,
    workspace_offset_for,
)
from sweep_bench.tasks.conveyor_base.randomization_scene import (
    build_clutter,
    build_texture_overlay,
    sample_camera_jitter,
    sample_clutter_count,
    sample_lighting_params,
    sample_table_material,
    sample_tint,
    scene_randomization_active,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.structs import Articulation, Link
    from mani_skill.utils.structs.actor import Actor

    from sweep_bench.assets.catalog import ObjectSpec


class _ParallelJawAgent(Protocol):
    finger1_link: Link
    finger2_link: Link
    robot: Articulation


def _resolve_objects(catalog: Sequence[ObjectSpec], requested: tuple[str, ...] | None) -> tuple[str, ...]:
    catalog_ids = tuple(spec.id for spec in catalog)
    if requested is None:
        return catalog_ids[:1]
    unknown = [o for o in requested if o not in catalog_ids]
    if unknown:
        raise ValueError(f"Unknown objects {unknown}; choose from {sorted(catalog_ids)}")
    return tuple(requested)


_BACKGROUND_STREAM = 0xB6_C0


_PANDA_UIDS = ("panda", "panda_wristcam")


_PALM_LINK = {
    "panda": "panda_hand",
    "panda_wristcam": "panda_hand",
}

_FINGER_LINK_ZERO_OFFSET: dict[str, float] = {
    "panda": 0.0,
    "panda_wristcam": 0.0,
    "fetch": 0.03085,
}


Aperture = float | np.ndarray | torch.Tensor


_LEAKED_RENDER_MATERIALS: list[sapien.render.RenderMaterial] = []


def _leak_render_materials(scene: ManiSkillScene) -> None:
    for sub_scene in scene.sub_scenes:
        for entity in sub_scene.entities:
            for comp in entity.components:
                if isinstance(comp, sapien.render.RenderBodyComponent):
                    for shape in comp.render_shapes:
                        _LEAKED_RENDER_MATERIALS.extend(part.material for part in shape.parts)


_CONTEXT_SLOTS = 3

_BELT_SLOTS = 2


def _pose_row(actor: Actor) -> torch.Tensor:
    pose = actor.pose
    return torch.cat([pose.p, pose.q], dim=-1)


class ConveyorWorkspaceEnv(BaseEnv):
    SUPPORTED_ROBOTS = [
        "panda",
        "panda_wristcam",
    ]

    robot_uids: str

    default_prompt: str

    RandomizationConfig: ClassVar[type]

    randomization: ConveyorMotionRandomization

    SUPPORTED_FRUITS: ClassVar[tuple[str, ...]] = ()

    OBJECT_CATALOG: ClassVar[tuple[ObjectSpec, ...]] = ()

    conveyor_speed: float = 0.10

    belt_static_friction: float = 0.6
    belt_kinetic_friction: float = 0.4

    default_control_mode: str = "pd_joint_pos"

    _platform_half_size = (0.7, 1.05, 0.4)
    _platform_center_dx = -0.3

    _conveyor_half_length: float = 0.9
    _conveyor_half_width: float = 0.08
    _conveyor_deck_height: float = 0.25

    _rest_qpos = (0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4)

    _panda_base_dx: float = -0.615

    _BELT_ACCEL_SUPPORTED: ClassVar[bool] = True
    _CARRIED_RADIUS: ClassVar[float] = 0.12

    VelocityRandomizationConfig = VelocityRandomizationConfig

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "_initialize_episode" in cls.__dict__:
            raise TypeError(
                f"{cls.__module__}.{cls.__qualname__} overrides "
                f"`_initialize_episode`, which owns the per-env episode-RNG "
                f"swap. Override `_initialize_episode_draws` instead (call "
                f"`super()._initialize_episode_draws(...)` to keep the shared "
                f"belt/object/rest-pose init)."
            )

    def __init__(
        self,
        *args,
        prompt: str | None = None,
        control_mode: str | None = None,
        scene_style: SceneStyle = "replicacad",
        platform_style: PlatformStyle = "box",
        replicacad_scene_idx: int = 0,
        ai2thor_scene_idx: int = 0,
        robocasa_scene_idx: int = 0,
        workspace_offset: tuple[float, float, float] | None = None,
        background_pool: Sequence[Background] | None = None,
        shader_pack: str = "default",
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        randomization: ConveyorMotionRandomization | None = None,
        velocity_randomization: VelocityRandomizationConfig | None = None,
        objects: tuple[str, ...] | None = None,
        robot_uids: str = "panda_wristcam",
        early_fail: bool = False,
        record_contact: bool = False,
        **kwargs,
    ) -> None:
        require_servable_catalogs(type(self))
        if early_fail and robot_uids not in ROBOT_REACH:
            raise ValueError(
                f"early failure needs a measured reach bound for the robot, and {robot_uids!r} has none "
                f"(measured: {sorted(ROBOT_REACH)})"
            )
        self.early_fail: bool = early_fail
        self._early_fail_step: torch.Tensor | None = None
        self._early_fail_carry: torch.Tensor | None = None
        self.record_contact: bool = record_contact
        self._contact_record: ContactRecord | None = None
        self._hull_cache: dict[int, np.ndarray] = {}
        self._actor_hull_cache: dict[int, torch.Tensor] = {}
        if shader_pack.startswith("rt") and num_envs > 1:
            raise ValueError(
                f"Ray-traced shader_pack={shader_pack!r} requires num_envs=1 "
                f"(got {num_envs}); ManiSkill cannot ray-trace parallel envs in one process."
            )
        self.prompt: str = prompt if prompt is not None else self.default_prompt
        if control_mode is None:
            control_mode = self.default_control_mode
        self.scene_style: SceneStyle = scene_style
        if platform_style not in ("box", "table"):
            raise ValueError(f"platform_style must be 'box' or 'table', got {platform_style!r}")
        self.platform_style: PlatformStyle = platform_style
        self.shader_pack: str = shader_pack
        self.replicacad_scene_idx: int = replicacad_scene_idx
        self.ai2thor_scene_idx: int = ai2thor_scene_idx
        self.robocasa_scene_idx: int = robocasa_scene_idx
        explicit_offset = workspace_offset is not None
        if workspace_offset is None:
            if scene_style == "replicacad":
                workspace_offset = workspace_offset_for(scene_style, replicacad_scene_idx)
            elif scene_style == "ai2thor":
                workspace_offset = workspace_offset_for(scene_style, ai2thor_scene_idx)
            elif scene_style == "robocasa":
                workspace_offset = workspace_offset_for(scene_style, robocasa_scene_idx)
            else:
                workspace_offset = (0.0, 0.0, 0.0)
        self.workspace_offset: tuple[float, float, float] = workspace_offset
        if background_pool is not None and explicit_offset:
            raise ValueError("background_pool places each background at its recorded spot; drop workspace_offset")
        self.background_pool: tuple[Background, ...] | None = (
            None if background_pool is None else validate_background_pool(background_pool)
        )
        self._background_draws = 0
        self.randomization = randomization if randomization is not None else self.RandomizationConfig()
        self.velocity_randomization: VelocityRandomizationConfig = (
            velocity_randomization if velocity_randomization is not None else self.VelocityRandomizationConfig()
        )
        self.objects: tuple[str, ...] = _resolve_objects(self.OBJECT_CATALOG, objects)
        self._eval_last_step: dict[str, torch.Tensor] = {}
        if reconfiguration_freq is None:
            scene_dr = randomization is not None and scene_randomization_active(randomization)
            varies_objects = objects is not None and len(objects) > 1
            reconfiguration_freq = 1 if (num_envs == 1 and (varies_objects or scene_dr)) else 0
        kwargs.setdefault("enhanced_determinism", True)
        super().__init__(
            *args,
            control_mode=control_mode,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            robot_uids=robot_uids,
            **kwargs,
        )

    def built_object_ids(self) -> tuple[str, ...]:
        return self.objects

    def _clear(self) -> None:
        if platform.system() == "Darwin" and self.scene is not None:
            _leak_render_materials(self.scene)
        super()._clear()

    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=20,
            gpu_memory_config=GPUMemoryConfig(collision_stack_size=2**23),
        )

    @property
    def _default_human_render_camera_configs(self):
        return human_render_camera_config(self.workspace_offset, self.shader_pack)

    @property
    def _default_viewer_camera_configs(self):
        return viewer_camera_config(self.shader_pack)

    @property
    def _default_sensor_configs(self):
        eye_jitters: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),) * len(EXTERIOR_RIG)
        if self.randomization.camera_jitter:
            eye_jitters = sample_camera_jitter(self._episode_rng, self.randomization)
        return sensor_camera_configs(
            self.workspace_offset,
            self.shader_pack,
            self.agent,
            eye_jitters=eye_jitters,
        )

    def _load_lighting(self, options: dict):
        intensity_scale = 1.0
        color_scale = (1.0, 1.0, 1.0)
        key_dir_offset = (0.0, 0.0, 0.0)
        if self.randomization.lighting:
            intensity_scale, color_scale, key_dir_offset = sample_lighting_params(self._episode_rng, self.randomization)
        build_lighting(
            self.scene,
            self.scene_style,
            self.workspace_offset,
            intensity_scale=intensity_scale,
            color_scale=color_scale,
            key_dir_offset=key_dir_offset,
        )

    def _load_agent(
        self, options: dict, initial_agent_poses: sapien.Pose | Pose | None = None, build_separate: bool = False
    ) -> None:
        if self.scene_style == "robocasa" and initial_agent_poses is None:
            initial_agent_poses = Pose.create_from_pq(p=torch.zeros((self.num_envs, 3), device=self.device))
        super()._load_agent(options, initial_agent_poses, build_separate)

    @property
    def background(self) -> Background:
        if self.scene_style == "replicacad":
            return ("replicacad", self.replicacad_scene_idx)
        if self.scene_style == "ai2thor":
            return ("ai2thor", self.ai2thor_scene_idx)
        if self.scene_style == "robocasa":
            return ("robocasa", self.robocasa_scene_idx)
        return ("table", 0)

    def _set_main_rng(self, seed) -> None:
        if seed is not None or self._main_seed is None:
            self._background_draws = 0
        super()._set_main_rng(seed)

    def _reconfigure(self, options: dict | None = None) -> None:
        if self.background_pool is not None:
            self._use_background(self._draw_background())
        self._hull_cache = {}
        self._actor_hull_cache = {}
        super()._reconfigure({} if options is None else options)

    def _draw_background(self) -> Background:
        assert self.background_pool is not None
        stream = np.random.default_rng([_BACKGROUND_STREAM, int(self._main_seed[0]), self._background_draws])
        self._background_draws += 1
        return self.background_pool[int(stream.integers(len(self.background_pool)))]

    def _use_background(self, background: Background) -> None:
        style, idx = background
        self.scene_style = style
        if style == "replicacad":
            self.replicacad_scene_idx = idx
        elif style == "ai2thor":
            self.ai2thor_scene_idx = idx
        elif style == "robocasa":
            self.robocasa_scene_idx = idx
        self.workspace_offset = (0.0, 0.0, 0.0) if style == "table" else workspace_offset_for(style, idx)

    def _load_scene(self, options: dict):
        tint, table_material = self._draw_platform_look()

        if self.scene_style == "replicacad":
            self.scene_builder = ReplicaCADSceneBuilder(self, include_staging_scenes=True)
            self.scene_builder.build(self.replicacad_scene_idx)
            self.table_scene = None
        elif self.scene_style == "ai2thor":
            self.scene_builder = ArchitecTHORSceneBuilder(self)
            self.scene_builder.build(self.ai2thor_scene_idx)
            self.table_scene = None
        elif self.scene_style == "robocasa":
            self.scene_builder = RoboCasaSceneBuilder(self)
            self._build_robocasa_layout(self.scene_builder)
            self.table_scene = None
        else:
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()
            self.scene_builder = None

        ox, oy, oz = self.workspace_offset

        if self.scene_style == "table" and table_material is not None:
            self.platform = None
            self.texture_overlay = build_textured_overlay(
                self.scene,
                top_center=(ox + self._platform_center_dx, oy, oz),
                half_size=(self._platform_half_size[0], self._platform_half_size[1]),
                material=table_render_material(table_material),
                tile_m=load_table_materials().get(table_material).tile_m,
                name="texture_overlay",
            )
        elif self.scene_style == "table":
            self.platform = None
            self.texture_overlay = (
                build_texture_overlay(
                    self.scene,
                    tint,
                    workspace_offset=(ox, oy, oz),
                    half_size=(self._platform_half_size[0], self._platform_half_size[1]),
                    center_dx=self._platform_center_dx,
                )
                if tint is not None
                else None
            )
        elif table_material is not None:
            phx, phy, phz = self._platform_half_size
            self.platform = build_table_platform(
                self.scene,
                center=(ox + self._platform_center_dx, oy, oz - phz),
                half_size=(phx, phy, phz),
                top_material=table_render_material(table_material),
                tile_m=load_table_materials().get(table_material).tile_m,
                name="workspace_platform",
            )
        else:
            phx, phy, phz = self._platform_half_size
            self.platform = build_box(
                self.scene,
                half_sizes=[phx, phy, phz],
                color=tint if tint is not None else [0.85, 0.75, 0.6, 1.0],
                name="workspace_platform",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[ox + self._platform_center_dx, oy, oz - phz]),
            )

        self.belt = build_conveyor(
            self.scene,
            workspace_offset=(ox, oy, oz),
            deck_height=self._conveyor_deck_height,
            belt_half_width=self._conveyor_half_width,
            half_length=self._conveyor_half_length,
            static_friction=self.belt_static_friction,
            kinetic_friction=self.belt_kinetic_friction,
            name="conveyor",
        )

        n_clutter = sample_clutter_count(self._episode_rng, self.randomization) if self.randomization.clutter else 0
        self.clutter = (
            build_clutter(
                self.scene,
                self._episode_rng,
                n_clutter,
                workspace_offset=(ox, oy, oz),
                belt_half_width=self._conveyor_half_width,
                half_length=self._conveyor_half_length,
            )
            if n_clutter
            else []
        )

        self._load_objects(options)

    def _draw_platform_look(self) -> tuple[list[float] | None, str | None]:
        tint: list[float] | None = None
        material: str | None = None
        if self.platform_style == "table":
            manifest = load_table_materials()
            if self.randomization.texture:
                material = sample_table_material(self._episode_rng, manifest.ids)
            elif self.scene_style != "table":
                material = manifest.default
        elif self.randomization.texture:
            tint = sample_tint(self._episode_rng)
        self._platform_material: np.ndarray | None = None if material is None else np.full(self.num_envs, material)
        return tint, material

    def _load_objects(self, options: dict) -> None:
        raise NotImplementedError

    @contextlib.contextmanager
    def _episode_rngs_set_aside(self) -> Iterator[None]:
        episode_rng, batched = self._episode_rng, self._batched_episode_rng
        self._batched_episode_rng = BatchedRNG.from_seeds([0] * self.num_envs)
        self._episode_rng = self._batched_episode_rng[0]
        try:
            yield
        finally:
            self._episode_rng, self._batched_episode_rng = episode_rng, batched

    def _build_robocasa_layout(self, builder: RoboCasaSceneBuilder) -> None:
        episode_rng = self._batched_episode_rng
        self._batched_episode_rng = BatchedRNG.from_seeds([self.robocasa_scene_idx] * self.num_envs)
        try:
            builder.build([self.robocasa_scene_idx] * self.num_envs)
        finally:
            self._batched_episode_rng = episode_rng

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        saved = self._episode_rng
        self._episode_rng = PerEnvEpisodeRNG([self._batched_episode_rng[int(i)] for i in env_idx])
        try:
            self._initialize_episode_draws(env_idx, options)
        finally:
            self._episode_rng = saved
        if self._early_fail_step is not None:
            self._early_fail_step[env_idx.to(self._early_fail_step.device)] = -1
        if self._contact_record is not None:
            self._contact_record.clear(env_idx)

    def _initialize_episode_draws(self, env_idx: torch.Tensor, options: dict) -> None:
        env_idx = env_idx.to(self.belt.velocity.device)
        self._align_elapsed_steps_with_sim()
        for last_step in self._eval_last_step.values():
            last_step[env_idx] = 0
        if self.scene_style in ("replicacad", "ai2thor"):
            self._initialize_scene_builder_workspace(env_idx)
        elif self.scene_style == "robocasa":
            self._place_robot_at_workspace(env_idx)
        else:
            assert self.table_scene is not None
            if self.background_pool is None:
                self.table_scene.initialize(env_idx)
            else:
                with self._episode_rngs_set_aside():
                    self.table_scene.initialize(env_idx)
            self._reset_rest_qpos(env_idx)

        b = len(env_idx)
        self.belt.reset(env_idx, self._setup_conveyor_motion(env_idx, b))
        self._initialize_objects(env_idx, b)
        self._latch_belt_entry(env_idx, b)
        self._start_belt_accel(env_idx)

    def _align_elapsed_steps_with_sim(self) -> None:
        sim_device = self.belt.velocity.device
        if self._elapsed_steps.device != sim_device:
            self._elapsed_steps = self._elapsed_steps.to(sim_device)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        raise NotImplementedError

    _belt_entry_mask: torch.Tensor

    def _latch_belt_entry(self, env_idx: torch.Tensor, b: int) -> None:
        riders = self.belt.riders
        if getattr(self, "_belt_entry_mask", None) is None or self._belt_entry_mask.shape != (
            self.num_envs,
            len(riders),
        ):
            self._belt_entry_mask = torch.zeros((self.num_envs, len(riders)), dtype=torch.bool, device=self.device)
        if not riders:
            return
        mask = torch.stack([self._on_deck(actor.pose.p[env_idx]) for actor in riders], dim=-1)
        self._belt_entry_mask[env_idx] = mask

        phase = self._sample_spawn_phase(env_idx, b)
        if phase is None:
            return
        rider_y = torch.zeros((self.num_envs, len(riders)), device=self.device)
        rider_y[env_idx] = torch.stack([actor.pose.p[env_idx, 1] for actor in riders], dim=-1)

        def shift_for(p: torch.Tensor) -> torch.Tensor:
            shift = torch.zeros(self.num_envs, device=self.device)
            shift[env_idx] = phase * self._belt_entry_lane_sign(env_idx, p)
            return shift

        for i, actor in enumerate(riders):
            self._phase_actor(actor, self._belt_entry_mask[:, i], shift_for(actor.pose.p[env_idx]))
        main_shift = shift_for(torch.full((len(env_idx), 3), self.workspace_offset[0], device=self.device))
        for actor in self._belt_entry_companions():
            self._phase_actor(actor, self._rides_with_entry(actor, rider_y), main_shift)

    def _phase_actor(self, actor: Actor, moves: torch.Tensor, shift: torch.Tensor) -> None:
        idxs = actor._scene_idxs
        writable = self.scene._reset_mask[idxs]
        if not bool(writable.any()):
            return
        pose = actor.pose
        p = pose.p[writable].clone()
        p[:, 1] = p[:, 1] + torch.where(moves[idxs][writable], shift[idxs][writable], torch.zeros_like(p[:, 1]))
        actor.set_pose(Pose.create_from_pq(p, pose.q[writable]))

    _BELT_ENTRY_LATERAL_MARGIN: float = 0.06
    _BELT_ENTRY_Z_BAND: tuple[float, float] = (-0.05, 0.20)

    _BELT_ENTRY_COMPANION_GAP: float = 0.15

    def _on_deck(self, p: torch.Tensor) -> torch.Tensor:
        oz = self.workspace_offset[2]
        dz = p[:, 2] - (oz + self._conveyor_deck_height)
        lo, hi = self._BELT_ENTRY_Z_BAND
        return self._over_belt(p) & (dz >= lo) & (dz <= hi)

    def _belt_entry_lane_dx(self) -> tuple[float, ...]:
        return (0.0,)

    def _belt_entry_lane_sign(self, env_idx: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.belt.direction_sign[env_idx]

    def _over_belt(self, p: torch.Tensor) -> torch.Tensor:
        ox = self.workspace_offset[0]
        band = self._conveyor_half_width + self._BELT_ENTRY_LATERAL_MARGIN
        lanes = [torch.abs(p[:, 0] - (ox + dx)) <= band for dx in self._belt_entry_lane_dx()]
        return torch.stack(lanes, dim=-1).any(dim=-1)

    def _belt_entry_companions(self) -> list[Actor]:
        rider_entities = {id(entity) for actor in self.belt.riders for entity in actor._objs}
        return [
            actor
            for actor in self.scene.actors.values()
            if actor.px_body_type == "dynamic" and not any(id(entity) in rider_entities for entity in actor._objs)
        ]

    def _rides_with_entry(self, actor: Actor, rider_y: torch.Tensor) -> torch.Tensor:
        idxs = actor._scene_idxs
        p = actor.pose.p
        oz = self.workspace_offset[2]
        above_deck = p[:, 2] - (oz + self._conveyor_deck_height) >= self._BELT_ENTRY_Z_BAND[0]
        near = torch.abs(p[:, 1].unsqueeze(-1) - rider_y[idxs]) <= self._BELT_ENTRY_COMPANION_GAP
        rides = self._over_belt(p) & above_deck & (near & self._belt_entry_mask[idxs]).any(dim=-1)
        out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        out[idxs] = rides
        return out

    def _sample_spawn_phase(self, env_idx: torch.Tensor, b: int) -> torch.Tensor | None:
        span = self.randomization.spawn_phase_range
        if not math.isfinite(span):
            raise ValueError(f"spawn_phase_range must be finite, got {span}")
        if span < 0.0:
            raise ValueError(f"spawn_phase_range must be >= 0 (0 disables it), got {span}")
        if span == 0.0:
            return None
        draw = self._episode_rng.uniform(0.0, span, size=b).astype(np.float32)
        return torch.as_tensor(draw, device=self.device)

    def rider_in_onset_view(self) -> torch.Tensor:
        mask = self._belt_entry_mask
        riders = self.belt.riders
        if not riders:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        arriving_in_view = mask & torch.stack([self._in_exterior_frustum(actor.pose.p) for actor in riders], dim=-1)
        seen = arriving_in_view.any(dim=-1)
        target = self._onset_target_column()
        if target is not None:
            seen = torch.where(mask[:, target], arriving_in_view[:, target], seen)
        return seen | ~mask.any(dim=-1)

    def _onset_target_column(self) -> int | None:
        target = self._primary_rider
        return next((i for i, actor in enumerate(self.belt.riders) if actor is target), None)

    def _in_exterior_frustum(self, p: torch.Tensor) -> torch.Tensor:
        offset = torch.as_tensor(self.workspace_offset, dtype=torch.float32, device=self.device)
        seen = torch.zeros(p.shape[0], dtype=torch.bool, device=self.device)
        for _uid, nominal_eye, nominal_target, fov in EXTERIOR_RIG:
            eye = torch.as_tensor(nominal_eye, dtype=torch.float32, device=self.device) + offset
            target = torch.as_tensor(nominal_target, dtype=torch.float32, device=self.device) + offset
            forward = target - eye
            forward = forward / torch.linalg.norm(forward)
            right = torch.linalg.cross(forward, torch.tensor([0.0, 0.0, 1.0], device=self.device))
            right = right / torch.linalg.norm(right)
            up = torch.linalg.cross(right, forward)

            v = p - eye
            depth = v @ forward
            limit = math.tan(fov / 2.0) * depth
            seen |= (depth > 0.0) & (torch.abs(v @ right) <= limit) & (torch.abs(v @ up) <= limit)
        return seen

    @property
    def _primary_rider(self) -> Actor:
        return self.belt.primary_rider

    @property
    def _primary_rider_pose(self) -> torch.Tensor:
        pose = self._primary_rider.pose
        return torch.cat([pose.p, pose.q], dim=-1)

    @property
    def _task_actors(self) -> list[Actor]:
        return self.belt.riders

    @property
    def _context_actors(self) -> list[Actor]:
        primary = self._primary_rider
        return [actor for actor in self._task_actors if actor is not primary]

    @property
    def _velocity_belts(self) -> list[ConveyorBelt]:
        return [self.belt]

    def _task_actor_poses(self) -> list[torch.Tensor]:
        rows = [self._primary_rider_pose]
        rows.extend(_pose_row(actor) for actor in self._context_actors)
        return rows

    def _lead_ordered(self, rows: list[torch.Tensor], leads: torch.Tensor) -> list[torch.Tensor]:
        if len(rows) < 2:
            raise RuntimeError(
                f"{type(self).__name__} asked for slot ordering with {len(rows)} reported pose(s); "
                f"it takes a PAIR of riders to order."
            )
        lead = leads.reshape(-1, 1)
        first, second = rows[0], rows[1]
        return [torch.where(lead, first, second), torch.where(lead, second, first), *rows[2:]]

    def _get_obs_extra(self, info: dict) -> dict:
        agent = cast("Panda", self.agent)
        rows = self._task_actor_poses()
        rider, context = rows[0], rows[1:]
        if len(context) > _CONTEXT_SLOTS:
            raise RuntimeError(
                f"{type(self).__name__} reports {len(context)} context poses but the shared "
                f"observation carries {_CONTEXT_SLOTS}; drop the actors the success predicate does not "
                f"read, or widen _CONTEXT_SLOTS for the whole suite."
            )
        slots = [*context, *([rider] * (_CONTEXT_SLOTS - len(context)))]
        belts = self._velocity_belts
        if len(belts) > _BELT_SLOTS:
            raise RuntimeError(
                f"{type(self).__name__} declares {len(belts)} velocity belts but the shared observation "
                f"carries {_BELT_SLOTS}; drop the ones derivable from the first, or widen _BELT_SLOTS."
            )
        _belt_columns = [belt.velocity.reshape(-1, 1) for belt in belts]
        _belt_columns += [_belt_columns[0]] * (_BELT_SLOTS - len(_belt_columns))
        return dict(
            tcp_pose=torch.cat([agent.tcp.pose.p, agent.tcp.pose.q], dim=-1),
            rider_pose=rider,
            context_poses=torch.cat(slots, dim=-1),
            belt_velocity=torch.cat(_belt_columns, dim=-1),
        )

    def belt_carried(self, actor: Actor, frac: float) -> torch.Tensor:
        return belt_carries(actor.linear_velocity[:, 1], self.belt.velocity, frac)

    def riding_on_deck(self, actor: Actor, rest_height: float, max_rise: float = DEFAULT_MAX_RISE) -> torch.Tensor:
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        return rider_at_rest_height(actor.pose, deck_z=deck_z, rest_height=rest_height, max_rise=max_rise)

    @property
    def _belt_push_watch(self) -> Actor | None:
        return None

    def _reset_belt_push_window(self) -> None:
        if self._belt_push_watch is None:
            return
        if not hasattr(self, "_belt_push_hi"):
            self._belt_push_hi = torch.zeros(self.num_envs, device=self.device)
        self._belt_push_hi[:] = 0.0

    def _accumulate_belt_push(self) -> None:
        actor = self._belt_push_watch
        if actor is None or not hasattr(self, "_belt_push_hi"):
            return
        self._belt_push_hi = torch.maximum(self._belt_push_hi, self._belt_push_sample(actor))

    def _belt_push_sample(self, actor: Actor) -> torch.Tensor:
        agent = cast("_ParallelJawAgent", self.agent)
        palm_name = _PALM_LINK.get(self.robot_uids)
        if palm_name is None:
            raise ValueError(
                f"belt provenance has no gripper palm link for robot_uids={self.robot_uids!r}; "
                f"known: {sorted(_PALM_LINK)}"
            )
        total = torch.zeros((self.num_envs, 3), device=self.device)
        for link in (agent.finger1_link, agent.finger2_link, agent.robot.links_map[palm_name]):
            total = total + self.scene.get_pairwise_contact_forces(link, actor)
        return -(total[:, 1] * self.belt.direction_sign)

    def gripper_drove_along_belt(self, force_min: float = 0.5) -> torch.Tensor:
        if not hasattr(self, "_belt_push_hi"):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._belt_push_hi > force_min

    def gripper_aperture(self) -> torch.Tensor:
        offset = _FINGER_LINK_ZERO_OFFSET.get(self.robot_uids)
        if offset is None:
            raise ValueError(
                f"gripper_aperture has no finger-link offset for robot_uids={self.robot_uids!r}; "
                f"known: {sorted(_FINGER_LINK_ZERO_OFFSET)}"
            )
        agent = cast("_ParallelJawAgent", self.agent)
        separation = torch.linalg.norm(agent.finger1_link.pose.p - agent.finger2_link.pose.p, dim=-1)
        return separation - offset

    @overload
    def _gripper_action_for_aperture(self, aperture: float) -> float: ...
    @overload
    def _gripper_action_for_aperture(self, aperture: torch.Tensor) -> torch.Tensor: ...
    def _gripper_action_for_aperture(self, aperture: Aperture) -> Aperture:
        if self.robot_uids in _PANDA_UIDS:
            return (0.5 * aperture - 0.015) / 0.025
        raise ValueError(f"_gripper_action_for_aperture does not support robot_uids={self.robot_uids!r}")

    def _rest_aperture(self, env_idx: torch.Tensor) -> float | np.ndarray:
        return 0.08

    def _rest_aperture_values(self, env_idx: torch.Tensor, b: int) -> np.ndarray:
        return np.broadcast_to(np.asarray(self._rest_aperture(env_idx)).reshape(-1), (b,))

    def _rest_gripper_qpos(self, env_idx: torch.Tensor, b: int) -> np.ndarray:
        aperture = self._rest_aperture_values(env_idx, b)
        if self.robot_uids in _PANDA_UIDS:
            half = 0.5 * aperture
            return np.stack([half, half], axis=-1)
        raise ValueError(f"_rest_gripper_qpos does not support robot_uids={self.robot_uids!r}")

    def _rest_arm_qpos(self) -> tuple[float, ...]:
        if self.robot_uids in _PANDA_UIDS:
            return self._rest_qpos
        raise ValueError(f"_rest_arm_qpos does not support robot_uids={self.robot_uids!r}")

    def _base_dx(self) -> float:
        if self.robot_uids in _PANDA_UIDS:
            return self._panda_base_dx
        raise ValueError(f"_base_dx does not support robot_uids={self.robot_uids!r}")

    def _reset_rest_qpos(self, env_idx: torch.Tensor) -> None:
        b = len(env_idx)
        arm_rest = self._rest_arm_qpos()
        gripper_rest = self._rest_gripper_qpos(env_idx, b)
        rest = np.concatenate([np.asarray(arm_rest, dtype=np.float64), np.zeros(gripper_rest.shape[1])])
        qpos_np = self._episode_rng.normal(0, 0.02, (b, len(rest))) + rest
        qpos_np[:, len(arm_rest) :] = gripper_rest
        self.agent.reset(torch.as_tensor(qpos_np, dtype=torch.float32, device=self.device))

    def _setup_conveyor_motion(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        rcfg = self.randomization
        if rcfg.speed:
            lo, hi = rcfg.speed_range
            speeds = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        else:
            speeds = np.full(b, self.conveyor_speed, dtype=np.float32)

        if rcfg.direction:
            choices = np.array(
                [1.0 if d == "+y" else -1.0 for d in rcfg.directions],
                dtype=np.float32,
            )
            signs = self._episode_rng.choice(choices, size=b)
        else:
            signs = np.ones(b, dtype=np.float32)
        if self._belt_accel_on:
            speeds = self._belt_accel_entry_speeds(env_idx, speeds)
        return torch.as_tensor(speeds * signs, device=self.device)

    @property
    def _belt_accel_on(self) -> bool:
        rcfg = self.randomization
        return getattr(rcfg, "accel", 0.0) != 0.0 or tuple(getattr(rcfg, "accel_range", (0.0, 0.0))) != (0.0, 0.0)

    def _draw_belt_accels(self, b: int) -> np.ndarray:
        rcfg = self.randomization
        lo, hi = rcfg.accel_range
        if (lo, hi) == (0.0, 0.0):
            return np.full(b, rcfg.accel, dtype=np.float64)
        if rcfg.accel != 0.0:
            raise ValueError(f"set accel or accel_range, not both (accel={rcfg.accel}, accel_range={rcfg.accel_range})")
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            raise ValueError(f"accel_range must be finite with lo <= hi, got {rcfg.accel_range}")
        return self._episode_rng.uniform(lo, hi, size=b)

    def _belt_accel_entry_speeds(self, env_idx: torch.Tensor, center_speeds: np.ndarray) -> np.ndarray:
        if not self._BELT_ACCEL_SUPPORTED:
            raise ValueError(f"{type(self).__name__} does not support a belt acceleration profile")
        rcfg = self.randomization
        accels = self._draw_belt_accels(len(center_speeds))
        if not (np.isfinite(accels).all() and math.isfinite(rcfg.accel_window) and rcfg.accel_window > 0.0):
            raise ValueError(f"need finite accel and accel_window > 0, got {accels}, {rcfg.accel_window}")
        slowest_sq = center_speeds.astype(np.float64) ** 2 - np.abs(accels) * rcfg.accel_window
        if (slowest_sq <= 0.0).any():
            worst = int(np.argmin(slowest_sq))
            raise ValueError(
                f"accel={accels[worst]} over a {rcfg.accel_window} m window stops the belt for centre speed "
                f"{float(center_speeds[worst]):.3f} m/s; v_c^2 must exceed |accel| * accel_window"
            )
        if getattr(self, "_belt_center_speed", None) is None:
            self._belt_center_speed = torch.zeros(self.num_envs, device=self.device)
            self._belt_accel_u = torch.zeros(self.num_envs, device=self.device)
            self._belt_accel = torch.zeros(self.num_envs, device=self.device)
        self._belt_center_speed[env_idx] = torch.as_tensor(center_speeds, device=self.device)
        self._belt_accel[env_idx] = torch.as_tensor(accels, dtype=torch.float32, device=self.device)
        entry_sq = center_speeds.astype(np.float64) ** 2 - accels * rcfg.accel_window
        return np.sqrt(entry_sq).astype(np.float32)

    def _belt_accel_speed(self, u: torch.Tensor, center_speed: torch.Tensor, accel: torch.Tensor) -> torch.Tensor:
        half = 0.5 * self.randomization.accel_window
        return (center_speed.square() + 2.0 * accel * u.clamp(-half, half)).sqrt()

    def _start_belt_accel(self, env_idx: torch.Tensor) -> None:
        if not self._belt_accel_on:
            return
        drive = self.belt.drive
        sign = drive.direction_sign[env_idx]
        u0 = sign * (self._primary_rider_pose[env_idx, 1] - self.workspace_offset[1])
        self._belt_accel_u[env_idx] = u0
        launch = drive.velocity[env_idx].abs()
        start = self._belt_accel_speed(u0, self._belt_center_speed[env_idx], self._belt_accel[env_idx])
        drive.velocity[env_idx] = sign * start
        drive.velocity_np = drive.velocity.cpu().numpy()
        ratio = start / launch
        for i, actor in enumerate(self.belt.riders):
            on_deck = self._belt_entry_mask[env_idx, i]
            vel = actor.linear_velocity[env_idx].clone()
            vel[:, 1] = torch.where(on_deck, vel[:, 1] * ratio, vel[:, 1])
            actor.set_linear_velocity(vel)
        inside = u0 > -0.5 * self.randomization.accel_window
        if not self.belt.riders or not bool(inside.any()):
            return
        full_ratio = torch.ones(self.num_envs, device=self.device)
        full_ratio[env_idx] = torch.where(inside, ratio, torch.ones_like(ratio))
        rider_y = torch.zeros((self.num_envs, len(self.belt.riders)), device=self.device)
        rider_y[env_idx] = torch.stack([actor.pose.p[env_idx, 1] for actor in self.belt.riders], dim=-1)
        for actor in self._belt_entry_companions():
            idxs = actor._scene_idxs
            resetting = self.scene._reset_mask[idxs]
            moves = self._rides_with_entry(actor, rider_y)[idxs] & resetting
            if not bool(moves.any()):
                continue
            vel = actor.linear_velocity.clone()
            vel[:, 1] = torch.where(moves, vel[:, 1] * full_ratio[idxs], vel[:, 1])
            rows = resetting if self.scene.gpu_sim_enabled else torch.ones_like(resetting)
            actor.set_linear_velocity(vel[rows])

    def _advance_belt_accel(self) -> None:
        if not self._belt_accel_on:
            return
        drive = self.belt.drive
        speed = self._belt_accel_speed(self._belt_accel_u, self._belt_center_speed, self._belt_accel)
        drive.velocity[:] = drive.direction_sign * speed
        drive.velocity_np = drive.velocity.cpu().numpy()
        self._belt_accel_u += speed * self.control_timestep

    def _initialize_scene_builder_workspace(self, env_idx: torch.Tensor) -> None:
        from mani_skill.utils.structs import Articulation

        assert isinstance(self.scene_builder, (ReplicaCADSceneBuilder, ArchitecTHORSceneBuilder)), (
            "called outside a replicacad/ai2thor style"
        )

        for obj, pose in self.scene_builder._default_object_poses:
            obj.set_pose(pose)
            if isinstance(obj, Articulation):
                obj.set_qpos(obj.qpos[0] * 0)
                obj.set_qvel(obj.qvel[0] * 0)

        if self.scene.gpu_sim_enabled and len(env_idx) == self.num_envs:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene.px.step()
            self.scene._gpu_fetch_all()

        self._place_robot_at_workspace(env_idx)

    def _place_robot_at_workspace(self, env_idx: torch.Tensor) -> None:
        self._reset_rest_qpos(env_idx)
        ox, oy, oz = self.workspace_offset
        self.agent.robot.set_pose(sapien.Pose(p=[ox + self._base_dx(), oy + 0.0, oz]))

    def _eval_advanced(self, key: str) -> torch.Tensor:
        last_step = self._eval_last_step.get(key)
        if last_step is None:
            last_step = torch.zeros_like(self.elapsed_steps)
            self._eval_last_step[key] = last_step
        advanced = self.elapsed_steps != last_step
        last_step.copy_(self.elapsed_steps)
        return advanced

    def _advance_sustain(self, key: str, counter: torch.Tensor, holding: torch.Tensor) -> torch.Tensor:
        return torch.where(holding, counter + self._eval_advanced(key).long(), torch.zeros_like(counter))

    _EARLY_FAIL_NEVER: ClassVar[str | None] = None
    _EARLY_FAIL_HELD_REACH: ClassVar[bool] = False

    def get_info(self) -> dict:
        info = super().get_info()
        if self.record_contact:
            self._record_contact()
        if not self.early_fail:
            return info
        if self._early_fail_step is None or self._early_fail_carry is None:
            self._early_fail_step = torch.full_like(self.elapsed_steps, -1)
            self._early_fail_carry = torch.zeros(self.num_envs, device=self.device)
        fresh = self.elapsed_steps == 0
        if self._EARLY_FAIL_HELD_REACH and bool(fresh.any()):
            self._early_fail_carry = torch.where(fresh, self._held_diameter(), self._early_fail_carry)
        if self._EARLY_FAIL_NEVER is None:
            fired = self._success_impossible(info) & ~info["success"] & ~fresh
        else:
            fired = torch.zeros_like(info["success"])
        step = self._early_fail_step
        step.copy_(torch.where(fired & (step < 0), self.elapsed_steps, step))
        info["early_fail_step"] = step.clone()
        info["fail"] = fired
        return info

    @property
    def early_fail_step(self) -> torch.Tensor | None:
        return self._early_fail_step

    def _success_impossible(self, info: dict) -> torch.Tensor:
        return self._departed(self._primary_rider)

    def _departed(self, actor: Actor) -> torch.Tensor:
        assert self._early_fail_carry is not None, "_departed runs inside get_info with early failure on"
        rows = actor._scene_idxs
        points = self._actor_hull(actor)
        pose = actor.pose
        world = quaternion_apply(pose.q[:, None, :].expand(-1, points.shape[1], -1), points) + pose.p[:, None, :]
        base = self.agent.robot.pose.p[rows, :2]
        reach = torch.linalg.norm(world[..., :2] - base[:, None, :], dim=-1).amin(dim=-1) - HULL_TOLERANCE
        downstream = self.belt.direction_sign[rows, None]
        along = ((world[..., 1] - self.workspace_offset[1]) * downstream).amin(dim=-1) - HULL_TOLERANCE
        rel = pose.p[:, :2] - base
        returning = (rel * actor.linear_velocity[:, :2]).sum(dim=-1) < -RETURN_SPEED * torch.linalg.norm(rel, dim=-1)
        out_of_reach = reach > ROBOT_REACH[self.robot_uids] + self._early_fail_carry[rows]
        gone = out_of_reach & (along > self._conveyor_half_length) & ~returning
        out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        out[rows] = gone
        return out & (self.belt.velocity != 0.0)

    def _held_diameter(self) -> torch.Tensor:
        tcp = cast("Panda", self.agent).tcp.pose.p
        held = torch.zeros(self.num_envs, device=self.device)
        for actor in self.scene.actors.values():
            if actor.px_body_type != "dynamic" or actor.hidden:
                continue
            rows = actor._scene_idxs
            near = torch.linalg.norm(actor.pose.p - tcp[rows], dim=-1) < self._CARRIED_RADIUS
            diameter = torch.tensor(
                [2.0 * float(np.linalg.norm(self._entity_hull(e), axis=-1).max()) for e in actor._objs],
                device=self.device,
            )
            held[rows] = torch.maximum(held[rows], torch.where(near, diameter + HULL_TOLERANCE, 0.0))
        return held

    def _entity_hull(self, sapien_entity: sapien.Entity) -> np.ndarray:
        cached = self._hull_cache.get(id(sapien_entity))
        if cached is None:
            cached = self._hull_cache[id(sapien_entity)] = hull_points(sapien_entity)
        return cached

    def _actor_hull(self, actor: Actor) -> torch.Tensor:
        cached = self._actor_hull_cache.get(id(actor))
        if cached is None:
            clouds = [self._entity_hull(entity) for entity in actor._objs]
            n = max(len(cloud) for cloud in clouds)
            padded = np.stack([np.concatenate([c, np.repeat(c[:1], n - len(c), axis=0)]) for c in clouds])
            cached = self._actor_hull_cache[id(actor)] = torch.as_tensor(padded, device=self.device)
        return cached

    @property
    def contact_record(self) -> ContactRecord | None:
        return self._contact_record

    @property
    def _primary_rider_per_env(self) -> list[tuple[Actor, torch.Tensor]]:
        return [(self._primary_rider, torch.ones(self.num_envs, dtype=torch.bool, device=self.device))]

    def _record_contact(self) -> None:
        if self._contact_record is None:
            self._contact_record = ContactRecord.unset(self.elapsed_steps)
        agent = cast("_ParallelJawAgent", self.agent)
        hand = (agent.finger1_link, agent.finger2_link, agent.robot.links_map[_PALM_LINK[self.robot_uids]])
        force = torch.zeros(self.num_envs, device=self.device)
        position = torch.full((self.num_envs, 3), torch.nan, device=self.device)
        for rider, chosen in self._primary_rider_per_env:
            rows = rider._scene_idxs
            here = chosen[rows]
            rider_force = torch.zeros(len(rows), device=self.device)
            for link in hand:
                rider_force = rider_force + torch.linalg.norm(
                    self.scene.get_pairwise_contact_forces(link, rider), dim=-1
                )
            force[rows] = torch.where(here, rider_force, force[rows])
            position[rows] = torch.where(here[:, None], rider.pose.p, position[rows])
        touching = (force > CONTACT_FORCE) & (self.elapsed_steps > 0)
        u = self.belt.direction_sign * (position[:, 1] - self.workspace_offset[1])
        tcp = cast("Panda", self.agent).tcp.pose.p
        dist = torch.linalg.norm(tcp - position, dim=-1)
        offset = self.belt.direction_sign * (tcp[:, 1] - position[:, 1])
        speed = self._velocity_belts[0].velocity.abs()
        self._contact_record.update(self.elapsed_steps, touching, speed, u, dist, offset)

    def _before_control_step(self) -> None:
        self._advance_belt_accel()
        self.belt.on_control_step()
        self._reset_belt_push_window()

    def _before_simulation_step(self) -> None:
        self.belt.on_simulation_step()

    def _after_simulation_step(self) -> None:
        super()._after_simulation_step()
        self._accumulate_belt_push()
