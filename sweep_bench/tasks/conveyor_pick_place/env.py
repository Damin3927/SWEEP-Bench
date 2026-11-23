from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.procedural import build_bin
from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.cameras import human_render_camera_config
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_pick_place.config import RandomizationConfig
from sweep_bench.tasks.conveyor_pick_place.scripted import ScriptedPickPlaceMixin
from sweep_bench.tasks.conveyor_pick_place.variants import CUBE_SPEC, PICK_PLACE_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_BIN_COLOR = (0.32, 0.36, 0.44, 1.0)


@bench_task("ConveyorPickPlace-v0")
@register_env("ConveyorPickPlace-v0", max_episode_steps=300)
class ConveyorPickPlaceEnv(ScriptedPickPlaceMixin, ConveyorWorkspaceEnv):
    """Pick a cube off the moving belt and place it into a static drop-off bin."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(PICK_PLACE_OBJECT_IDS, extra=(CUBE_SPEC,))

    _bin_inner_half_x: float = 0.06
    _bin_inner_half_y: float = 0.06
    _bin_wall_thickness: float = 0.008
    _bin_wall_half_height: float = 0.025

    @property
    def _bin_floor_half_height(self) -> float:
        return self._conveyor_deck_height / 2.0

    _bin_offset_x: float = -0.28
    _bin_offset_y_mag: float = 0.35
    _bin_fast_band_speed: float = 0.25

    _cube_patch_radius: float = 0.02

    _released_distance_threshold: float = 0.08
    _settled_speed_threshold: float = 0.10
    _place_sustain_steps: int = 5

    default_prompt: str = "pick up the cube from the conveyor belt and place it in the bin"

    @property
    def episode_objects(self) -> list[str]:
        return list(self._rider_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"pick up the {name} from the conveyor belt and place it in the bin" for name in self._rider_display]

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

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
        self._rider_names: list[str] = [str(name) for name in names]
        self._rider_display: list[str] = [display_name(spec_by_id[name]) for name in self._rider_names]
        rider_objs: list[Actor] = []
        patch_radii: list[float] = []
        half_zs: list[float] = []
        half_fps: list[float] = []
        for i, name in enumerate(self._rider_names):
            spec = spec_by_id[name]
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + spec.bbox[2] + 0.001])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"{name}-{i}")
            self.remove_from_state_dict_registry(obj)
            rider_objs.append(obj)
            patch_radii.append(spec.patch_radius if spec.patch_radius is not None else self._cube_patch_radius)
            half_zs.append(spec.bbox[2])
            half_fps.append(max(spec.bbox[0], spec.bbox[1]))
        self.cube = Actor.merge(rider_objs, name="pick_cube")
        self.add_to_state_dict_registry(self.cube)
        self.belt.register_rider(
            self.cube,
            patch_radius=torch.as_tensor(patch_radii, dtype=torch.float32, device=self.device),
        )
        self._rider_half_z = torch.as_tensor(half_zs, dtype=torch.float32, device=self.device)
        self._rider_half_fp = torch.as_tensor(half_fps, dtype=torch.float32, device=self.device)

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
        self.bin = bin_builder.build_kinematic(name="dropoff_bin")

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.bin]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        bin_pos = self._sample_bin_position(env_idx, b)
        bin_q = torch.zeros((b, 4), device=device)
        bin_q[:, 0] = 1.0
        self.bin.set_pose(Pose.create_from_pq(bin_pos, bin_q))

        cube_pos = self._sample_cube_position(env_idx, b)
        cube_q = torch.zeros((b, 4), device=device)
        cube_q[:, 0] = 1.0
        self.cube.set_pose(Pose.create_from_pq(cube_pos, cube_q))
        cube_vel = torch.zeros((b, 3), device=device)
        cube_vel[:, 1] = self.belt.velocity[env_idx]
        self.cube.set_linear_velocity(cube_vel)
        self.cube.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_place_trackers(env_idx)

    def _sample_cube_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._rider_half_fp[env_idx] - 0.04
        cube_pos = torch.zeros((b, 3), device=device)
        cube_pos[:, 0] = ox
        cube_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        cube_pos[:, 2] = deck_z + self._rider_half_z[env_idx] + 0.001
        return cube_pos

    def _sample_bin_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        device = self.device
        rcfg = self.randomization
        pos = torch.zeros((b, 3), device=device)
        if rcfg.bin_position:
            xs = self._episode_rng.uniform(*rcfg.bin_x_range, size=b).astype(np.float32)
            y_mag = self._episode_rng.uniform(*rcfg.bin_side_y_range, size=b).astype(np.float32)
            side = self._episode_rng.choice([-1.0, 1.0], size=b).astype(np.float32)
            pos[:, 0] = ox + torch.as_tensor(xs, device=device)
            pos[:, 1] = oy + torch.as_tensor(side * y_mag, device=device)
        else:
            speed = self.belt.velocity[env_idx].abs()
            fast_band = (speed >= self._bin_fast_band_speed) | (speed == 0.0)
            grasp_rel = torch.where(fast_band, 1.0, -1.0)
            pos[:, 0] = ox + self._bin_offset_x
            pos[:, 1] = oy + self._bin_offset_y_mag * grasp_rel * self.belt.direction_sign[env_idx]
        pos[:, 2] = oz + self._bin_origin_above_surface
        return pos

    def _reset_place_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_place_sustain_counter"):
            self._place_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._place_sustain_counter[env_idx] = 0

    def evaluate(self):
        agent = cast("Panda", self.agent)
        cube_pos = self.cube.pose.p
        bin_pos = self.bin.pose.p
        tcp_pos = agent.tcp.pose.p

        rel = cube_pos - bin_pos
        in_bin_xy = (torch.abs(rel[:, 0]) < self._bin_inner_half_x) & (torch.abs(rel[:, 1]) < self._bin_inner_half_y)
        z_lo = self._bin_floor_half_height
        z_hi = self._bin_floor_half_height + 2.0 * self._bin_wall_half_height
        in_bin_z = (rel[:, 2] > z_lo) & (rel[:, 2] < z_hi)

        cube_speed = torch.linalg.norm(self.cube.linear_velocity, dim=-1)
        settled = cube_speed < self._settled_speed_threshold
        released = (~agent.is_grasping(self.cube)) & (
            torch.linalg.norm(tcp_pos - cube_pos, dim=-1) > self._released_distance_threshold
        )

        placed = in_bin_xy & in_bin_z & settled & released

        self._place_sustain_counter = self._advance_sustain("pick_place", self._place_sustain_counter, placed)
        success = self._place_sustain_counter >= self._place_sustain_steps

        return {
            "success": success,
            "in_bin_xy": in_bin_xy,
            "in_bin_z": in_bin_z,
            "settled": settled,
            "released": released,
            "place_sustain": self._place_sustain_counter,
            "cube_to_bin_xy": torch.linalg.norm(rel[:, :2], dim=-1),
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        bin_pos = self.bin.pose.p

        tcp_to_cube = torch.linalg.norm(tcp_pos - cube_pos, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_cube)

        cube_to_bin_xy = torch.linalg.norm(cube_pos[:, :2] - bin_pos[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(3 * cube_to_bin_xy)

        in_bin = (info["in_bin_xy"] & info["in_bin_z"]).float()
        return reach_reward + carry_reward + 2.0 * in_bin

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0
