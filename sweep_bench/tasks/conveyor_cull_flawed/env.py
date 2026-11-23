from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.procedural import build_bin, build_box
from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.cameras import human_render_camera_config
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_cull_flawed.assets import add_defect_band
from sweep_bench.tasks.conveyor_cull_flawed.config import RandomizationConfig
from sweep_bench.tasks.conveyor_cull_flawed.scripted import ScriptedCullFlawedMixin
from sweep_bench.tasks.conveyor_cull_flawed.variants import CULL_FLAWED_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_BAND_COLORS: tuple[tuple[float, float, float, float], ...] = (
    (0.95, 0.10, 0.55, 1.0),
    (0.98, 0.42, 0.02, 1.0),
    (0.05, 0.90, 0.85, 1.0),
)
_BAND_RADIUS_FRACTIONS: tuple[float, ...] = (0.80, 0.60)
_BIN_COLOR = (0.55, 0.13, 0.13, 1.0)
_RESIDENT_COLOR = (0.42, 0.42, 0.45, 1.0)


@bench_task("ConveyorCullFlawed-v0")
@register_env("ConveyorCullFlawed-v0", max_episode_steps=300)
class ConveyorCullFlawedEnv(ScriptedCullFlawedMixin, ConveyorWorkspaceEnv):
    """Divert the FLAWED part into the reject bin; let a SOUND part ride past untouched."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(CULL_FLAWED_OBJECT_IDS)

    _bin_inner_half_x: float = 0.12
    _bin_inner_half_y: float = 0.07
    _bin_wall_thickness: float = 0.008
    _bin_wall_half_height: float = 0.03

    @property
    def _bin_floor_half_height(self) -> float:
        return self._conveyor_deck_height / 2.0

    _bin_offset_x: float = -0.28
    _bin_offset_y_mag: float = 0.35
    _bin_fast_band_speed: float = 0.25

    _resident_half_size: tuple[float, float, float] = (0.018, 0.018, 0.015)
    _resident_offset_x: float = 0.096
    _resident_disturb_tol: float = 0.02

    _part_patch_radius: float = 0.02

    _band_half_thickness: float = 0.002

    _bin_seat_tol: float = 0.004
    _settled_speed_threshold: float = 0.05
    _release_gap_margin: float = 0.012
    _release_distance: float = 0.10
    _cull_sustain_steps: int = 6

    _crossing_line: float = 0.40
    _deck_band: float = 0.020
    _lane_band: float = 0.050
    _standoff_distance: float = 0.12
    _finger_close_tol: float = 0.002

    _spawn_edge_margin: float = 0.04
    _park_offset: tuple[float, float, float] = (0.0, -1.35, -0.20)

    default_prompt: str = (
        "inspect the part on the conveyor belt: if it carries a defect band, put it in the reject bin; "
        "otherwise leave it alone and let it ride past"
    )

    @property
    def episode_objects(self) -> list[str]:
        return list(self._part_names)

    @property
    def episode_flaws(self) -> list[bool]:
        return [bool(v) for v in self._is_flawed.tolist()]

    @property
    def episode_prompts(self) -> list[str]:
        return [
            f"inspect the {name} on the conveyor belt: if it carries a defect band, put it in the reject bin; "
            f"otherwise leave it alone and let it ride past"
            for name in self._part_display
        ]

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        randomization: RandomizationConfig | None = None,
        reconfiguration_freq: int | None = None,
        num_envs: int = 1,
        objects: tuple[str, ...] | None = None,
        **kwargs,
    ) -> None:
        randomization = randomization if randomization is not None else self.RandomizationConfig()
        if reconfiguration_freq is None:
            band_dr = randomization.band_appearance
            reconfiguration_freq = (
                1 if (num_envs == 1 and (band_dr or (objects is not None and len(objects) > 1))) else 0
            )
        super().__init__(
            *args,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            randomization=randomization,
            objects=objects,
            **kwargs,
        )

    @property
    def _bin_outer_half_x(self) -> float:
        return self._bin_inner_half_x + 2.0 * self._bin_wall_thickness

    @property
    def _bin_outer_half_y(self) -> float:
        return self._bin_inner_half_y + 2.0 * self._bin_wall_thickness

    @property
    def _bin_origin_above_surface(self) -> float:
        return self._bin_floor_half_height

    _human_render_eye: tuple[float, float, float] = (0.75, 0.0, 1.25)
    _human_render_target: tuple[float, float, float] = (-0.1, 0.0, 0.15)

    @property
    def _default_human_render_camera_configs(self):
        return human_render_camera_config(
            self.workspace_offset,
            self.shader_pack,
            eye=self._human_render_eye,
            target=self._human_render_target,
        )

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._part_names: list[str] = [str(name) for name in names]
        self._part_display: list[str] = [display_name(spec_by_id[name]) for name in self._part_names]

        if self.randomization.band_appearance:
            band_colors = self._batched_episode_rng.choice(np.arange(len(_BAND_COLORS)))
            band_fracs = self._batched_episode_rng.choice(np.array(_BAND_RADIUS_FRACTIONS))
        else:
            band_colors = np.zeros(self.num_envs, dtype=np.int64)
            band_fracs = np.full(self.num_envs, _BAND_RADIUS_FRACTIONS[0], dtype=np.float32)

        flawed_objs: list[Actor] = []
        sound_objs: list[Actor] = []
        patch_radii: list[float] = []
        half_zs: list[float] = []
        half_fps: list[float] = []
        grasp_dias: list[float] = []
        for i, name in enumerate(self._part_names):
            spec = spec_by_id[name]
            for banded in (True, False):
                builder = spec.builder_fn(self.scene)
                if banded:
                    add_defect_band(
                        builder,
                        radius=float(band_fracs[i]) * min(spec.bbox[0], spec.bbox[1]),
                        half_thickness=self._band_half_thickness,
                        top_z=spec.bbox[2],
                        color=_BAND_COLORS[int(band_colors[i])],
                    )
                builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + spec.bbox[2] + 0.001])
                builder.set_scene_idxs([i])
                obj = builder.build(name=f"{name}-{'flawed' if banded else 'sound'}-{i}")
                self.remove_from_state_dict_registry(obj)
                (flawed_objs if banded else sound_objs).append(obj)
            patch_radii.append(spec.patch_radius if spec.patch_radius is not None else self._part_patch_radius)
            half_zs.append(spec.bbox[2])
            half_fps.append(max(spec.bbox[0], spec.bbox[1]))
            grasp_dias.append(
                spec.grasp_width if spec.grasp_width is not None else 2.0 * min(spec.bbox[0], spec.bbox[1])
            )
        self.part_flawed = Actor.merge(flawed_objs, name="flawed_part")
        self.part_sound = Actor.merge(sound_objs, name="sound_part")
        self.add_to_state_dict_registry(self.part_flawed)
        self.add_to_state_dict_registry(self.part_sound)
        patch = torch.as_tensor(patch_radii, dtype=torch.float32, device=self.device)
        self.belt.register_rider(self.part_flawed, patch_radius=patch)
        self.belt.register_rider(self.part_sound, patch_radius=patch)
        self._part_half_z = torch.as_tensor(half_zs, dtype=torch.float32, device=self.device)
        self._part_half_fp = torch.as_tensor(half_fps, dtype=torch.float32, device=self.device)
        self._part_grasp_dia = torch.as_tensor(grasp_dias, dtype=torch.float32, device=self.device)

        bin_builder = build_bin(
            self.scene,
            inner_half_x=self._bin_inner_half_x,
            inner_half_y=self._bin_inner_half_y,
            wall_thickness=self._bin_wall_thickness,
            wall_half_height=self._bin_wall_half_height,
            floor_half_height=self._bin_floor_half_height,
            color=_BIN_COLOR,
        )
        bin_builder.initial_pose = sapien.Pose(
            p=[ox + self._bin_offset_x, oy + self._bin_offset_y_mag, oz + self._bin_origin_above_surface]
        )
        self.reject_bin = bin_builder.build_kinematic(name="reject_bin")

        res_builder = build_box(
            self.scene,
            half_x=self._resident_half_size[0],
            half_y=self._resident_half_size[1],
            half_z=self._resident_half_size[2],
            color=_RESIDENT_COLOR,
        )
        res_builder.initial_pose = sapien.Pose(p=[ox + self._bin_offset_x, oy + self._bin_offset_y_mag, oz + 0.05])
        self.bin_resident = res_builder.build(name="culled_part")

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.reject_bin]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        oz = self.workspace_offset[2]
        self._reset_cull_trackers(env_idx)

        ident_q = torch.zeros((b, 4), device=device)
        ident_q[:, 0] = 1.0

        bin_pos = self._sample_bin_position(env_idx, b)
        self.reject_bin.set_pose(Pose.create_from_pq(bin_pos, ident_q))
        self._bin_pos[env_idx] = bin_pos

        res_pos = bin_pos.clone()
        res_pos[:, 0] = res_pos[:, 0] + self._resident_offset_x
        res_pos[:, 2] = bin_pos[:, 2] + self._bin_floor_half_height + self._resident_half_size[2]
        self.bin_resident.set_pose(Pose.create_from_pq(res_pos, ident_q))
        self.bin_resident.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.bin_resident.set_angular_velocity(torch.zeros((b, 3), device=device))
        self._resident_home[env_idx] = res_pos

        rcfg = self.randomization
        rate = self._episode_rng.uniform(*rcfg.flaw_rate_range, size=b) if rcfg.flaw_rate else np.full(b, 0.5)
        flawed_np = self._episode_rng.uniform(0.0, 1.0, size=b) < rate
        flawed = torch.as_tensor(flawed_np, dtype=torch.bool, device=device)
        self._is_flawed[env_idx] = flawed

        spawn_pos = self._sample_part_position(env_idx, b)
        park_pos = torch.zeros((b, 3), device=device)
        park_pos[:, 0] = self.workspace_offset[0] + self._park_offset[0]
        park_pos[:, 1] = self.workspace_offset[1] + self._park_offset[1]
        park_pos[:, 2] = oz + self._park_offset[2]
        on_belt = flawed.unsqueeze(-1)
        ride_vel = torch.zeros((b, 3), device=device)
        ride_vel[:, 1] = self.belt.velocity[env_idx]
        zero_vel = torch.zeros((b, 3), device=device)
        for actor, is_ride in ((self.part_flawed, on_belt), (self.part_sound, ~on_belt)):
            actor.set_pose(Pose.create_from_pq(torch.where(is_ride, spawn_pos, park_pos), ident_q))
            actor.set_linear_velocity(torch.where(is_ride, ride_vel, zero_vel))
            actor.set_angular_velocity(zero_vel)
        self._spawn_x[env_idx] = spawn_pos[:, 0]
        self._part_spawn_q[env_idx] = ident_q

        rest_aperture = self._rest_aperture_values(env_idx, b)
        self._rest_finger_gap[env_idx] = torch.as_tensor(rest_aperture, dtype=torch.float32, device=device)

    def _sample_part_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        start_y = self._conveyor_half_length - self._part_half_fp[env_idx] - self._spawn_edge_margin
        pos = torch.zeros((b, 3), device=self.device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._part_half_z[env_idx] + 0.001
        return pos

    def _sample_bin_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        device = self.device
        rcfg = self.randomization

        if rcfg.bin_side:
            side = torch.as_tensor(self._episode_rng.choice([-1.0, 1.0], size=b).astype(np.float32), device=device)
        else:
            speed = self.belt.velocity[env_idx].abs()
            fast_band = (speed >= self._bin_fast_band_speed) | (speed == 0.0)
            side = torch.where(fast_band, 1.0, -1.0) * self.belt.direction_sign[env_idx]

        if rcfg.bin_jitter:
            dx = torch.as_tensor(
                self._episode_rng.uniform(*rcfg.bin_jitter_x_range, size=b).astype(np.float32), device=device
            )
            dy = torch.as_tensor(
                self._episode_rng.uniform(*rcfg.bin_jitter_y_range, size=b).astype(np.float32), device=device
            )
        else:
            dx = torch.zeros(b, device=device)
            dy = torch.zeros(b, device=device)

        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox + self._bin_offset_x + dx
        pos[:, 1] = oy + (self._bin_offset_y_mag + dy) * side
        pos[:, 2] = oz + self._bin_origin_above_surface
        return pos

    def _reset_cull_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_cull_sustain_counter"):
            b, device = self.num_envs, self.device
            self._cull_sustain_counter = torch.zeros(b, dtype=torch.long, device=device)
            self._is_flawed = torch.zeros(b, dtype=torch.bool, device=device)
            self._bin_pos = torch.zeros((b, 3), device=device)
            self._resident_home = torch.zeros((b, 3), device=device)
            self._spawn_x = torch.zeros(b, device=device)
            self._part_spawn_q = torch.zeros((b, 4), device=device)
            self._part_spawn_q[:, 0] = 1.0
            self._rest_finger_gap = torch.zeros(b, device=device)
            self._reached_line = torch.zeros(b, dtype=torch.bool, device=device)
            self._crossed_clean = torch.zeros(b, dtype=torch.bool, device=device)
            self._approached = torch.zeros(b, dtype=torch.bool, device=device)
            self._bin_disturbed = torch.zeros(b, dtype=torch.bool, device=device)
        self._cull_sustain_counter[env_idx] = 0
        self._reached_line[env_idx] = False
        self._crossed_clean[env_idx] = False
        self._approached[env_idx] = False
        self._bin_disturbed[env_idx] = False

    def _finger_gap(self) -> torch.Tensor:
        return self.gripper_aperture()

    def _part_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        flawed = self._is_flawed.unsqueeze(-1)
        pos = torch.where(flawed, self.part_flawed.pose.p, self.part_sound.pose.p)
        vel = torch.where(flawed, self.part_flawed.linear_velocity, self.part_sound.linear_velocity)
        return pos, vel

    def _part_angular_velocity(self) -> torch.Tensor:
        flawed = self._is_flawed.unsqueeze(-1)
        return torch.where(flawed, self.part_flawed.angular_velocity, self.part_sound.angular_velocity)

    @property
    def _context_actors(self) -> list[Actor]:
        return [self.reject_bin]

    @property
    def _primary_rider_pose(self) -> torch.Tensor:
        flawed = self._is_flawed.unsqueeze(-1)
        p = torch.where(flawed, self.part_flawed.pose.p, self.part_sound.pose.p)
        q = torch.where(flawed, self.part_flawed.pose.q, self.part_sound.pose.q)
        return torch.cat([p, q], dim=-1)

    @property
    def _primary_rider_per_env(self) -> list[tuple[Actor, torch.Tensor]]:
        return [(self.part_flawed, self._is_flawed), (self.part_sound, ~self._is_flawed)]

    def _sound_branch_delivered(self, part_pos: torch.Tensor) -> torch.Tensor:
        oy = self.workspace_offset[1]
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        progress = (part_pos[:, 1] - oy) * self.belt.direction_sign
        at_line = progress >= self._crossing_line
        on_deck = torch.abs(part_pos[:, 2] - self._part_half_z - deck_z) < self._deck_band
        in_lane = torch.abs(part_pos[:, 0] - self._spawn_x) <= self._lane_band
        self._crossed_clean = self._crossed_clean | (at_line & ~self._reached_line & on_deck & in_lane)
        self._reached_line = self._reached_line | at_line
        return self._crossed_clean

    def _success_impossible(self, info: dict) -> torch.Tensor:
        sound_lost = self._approached | (self._reached_line & ~self._crossed_clean)
        flawed_lost = self._departed(self.part_flawed)
        return self._bin_disturbed | torch.where(self._is_flawed, flawed_lost, sound_lost)

    def evaluate(self):
        part_pos, part_vel = self._part_state()
        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        finger_gap = self._finger_gap()
        tcp_to_part = torch.linalg.norm(tcp_pos - part_pos, dim=-1)

        resident_shift = torch.linalg.norm(self.bin_resident.pose.p - self._resident_home, dim=-1)
        self._bin_disturbed = self._bin_disturbed | (resident_shift > self._resident_disturb_tol)
        bin_intact = ~self._bin_disturbed

        fingers_closed = finger_gap < (self._rest_finger_gap - self._finger_close_tol)
        self._approached = self._approached | (tcp_to_part < self._standoff_distance) | fingers_closed
        untouched = ~self._approached

        transited = self._sound_branch_delivered(part_pos) & untouched

        rel = part_pos - self._bin_pos
        in_bin = (torch.abs(rel[:, 0]) < self._bin_inner_half_x) & (torch.abs(rel[:, 1]) < self._bin_inner_half_y)
        bin_floor_top = self._bin_pos[:, 2] + self._bin_floor_half_height
        seated = torch.abs(part_pos[:, 2] - self._part_half_z - bin_floor_top) < self._bin_seat_tol
        at_rest = torch.linalg.norm(part_vel, dim=-1) < self._settled_speed_threshold
        released = (finger_gap >= self._part_grasp_dia + self._release_gap_margin) & (
            tcp_to_part >= self._release_distance
        )
        culled_now = in_bin & seated & at_rest & released
        self._cull_sustain_counter = self._advance_sustain("cull_sustain", self._cull_sustain_counter, culled_now)
        culled = self._cull_sustain_counter >= self._cull_sustain_steps

        success = torch.where(self._is_flawed, culled, transited) & bin_intact

        return {
            "success": success,
            "is_flawed": self._is_flawed,
            "in_bin": in_bin,
            "seated": seated,
            "at_rest": at_rest,
            "released": released,
            "cull_sustain": self._cull_sustain_counter,
            "crossed_clean": self._crossed_clean,
            "untouched": untouched,
            "bin_intact": bin_intact,
            "part_to_bin_xy": torch.linalg.norm(rel[:, :2], dim=-1),
            "tcp_to_part": tcp_to_part,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        part_pos, _vel = self._part_state()

        tcp_to_part = torch.linalg.norm(tcp_pos - part_pos, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_part)
        carry_reward = 1 - torch.tanh(3 * info["part_to_bin_xy"])
        flawed_reward = reach_reward + carry_reward + 2.0 * info["in_bin"].float()

        clearance = (tcp_to_part / self._standoff_distance).clamp(max=1.0)
        sound_reward = 2.0 * clearance + 2.0 * info["crossed_clean"].float()

        return torch.where(self._is_flawed, flawed_reward, sound_reward)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0
