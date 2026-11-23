from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.cameras import human_render_camera_config
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_upright_bottle.config import RandomizationConfig
from sweep_bench.tasks.conveyor_upright_bottle.scripted import ScriptedUprightBottleMixin
from sweep_bench.tasks.conveyor_upright_bottle.variants import (
    BOTTLE_SPECS,
    UPRIGHT_BOTTLE_OBJECT_IDS,
)
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorUprightBottle-v0")
@register_env("ConveyorUprightBottle-v0", max_episode_steps=300)
class ConveyorUprightBottleEnv(ScriptedUprightBottleMixin, ConveyorWorkspaceEnv):
    """Stand a fallen bottle back up ON the moving belt, so it rides away upright."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(UPRIGHT_BOTTLE_OBJECT_IDS, extra=BOTTLE_SPECS)

    _rider_patch_radius: float = 0.02

    _upright_tol_deg: float = 10.0
    _deck_rest_tol: float = 0.006
    _comove_tol: float = 0.05
    _lateral_speed_tol: float = 0.05
    _release_gap_margin: float = 0.012
    _release_distance: float = 0.08
    _belt_band_half: float = 0.05
    _belt_run_y: float = 0.85
    _stand_sustain_steps: int = 12

    _human_render_eye: tuple[float, float, float] = (0.75, 0.0, 1.25)
    _human_render_target: tuple[float, float, float] = (-0.1, 0.0, 0.20)

    default_prompt: str = "stand the fallen bottle upright on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _default_human_render_camera_configs(self):
        return human_render_camera_config(
            self.workspace_offset,
            self.shader_pack,
            eye=self._human_render_eye,
            target=self._human_render_target,
        )

    @property
    def episode_objects(self) -> list[str]:
        return list(self._rider_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"stand the fallen {name} upright on the conveyor belt" for name in self._rider_display]

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._rider_names: list[str] = [str(name) for name in names]
        self._rider_display: list[str] = [display_name(spec_by_id[n]) for n in self._rider_names]
        rider_objs: list[Actor] = []
        patch_radii: list[float] = []
        half_longs: list[float] = []
        half_shorts: list[float] = []
        grasp_widths: list[float] = []
        for i, name in enumerate(self._rider_names):
            spec = spec_by_id[name]
            if not (spec.bbox[0] > spec.bbox[1] and spec.bbox[0] > spec.bbox[2]):
                raise ValueError(
                    f"upright-bottle rider {spec.id!r} must have its LONG axis on local x "
                    f"(bbox[0] strictly largest), got bbox={spec.bbox}"
                )
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + spec.bbox[2] + 0.001])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"{name}-{i}")
            self.remove_from_state_dict_registry(obj)
            rider_objs.append(obj)
            patch_radii.append(spec.patch_radius if spec.patch_radius is not None else self._rider_patch_radius)
            half_longs.append(spec.bbox[0])
            half_shorts.append(spec.bbox[2])
            grasp_widths.append(spec.grasp_width if spec.grasp_width is not None else 2 * spec.bbox[1])
        self.bottle = Actor.merge(rider_objs, name="bottle")
        self.add_to_state_dict_registry(self.bottle)
        self.belt.register_rider(
            self.bottle,
            patch_radius=torch.as_tensor(patch_radii, dtype=torch.float32, device=self.device),
        )

        def _tensor(values: list[float]) -> torch.Tensor:
            return torch.as_tensor(values, dtype=torch.float32, device=self.device)

        self._rider_half_long = _tensor(half_longs)
        self._rider_half_z = _tensor(half_shorts)
        self._rider_grasp_width = _tensor(grasp_widths)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        pos = self._sample_bottle_position(env_idx, b)
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = math.cos(0.25 * math.pi)
        q[:, 3] = math.sin(0.25 * math.pi)
        self.bottle.set_pose(Pose.create_from_pq(pos, q))
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        self.bottle.set_linear_velocity(vel)
        self.bottle.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_stand_trackers(env_idx)

    def _sample_bottle_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._rider_half_long[env_idx] - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.rider_lateral_offset:
            lo, hi = self.randomization.lateral_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = pos[:, 0] + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._rider_half_z[env_idx] + 0.001
        return pos

    def _reset_stand_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_stand_sustain_counter"):
            self._stand_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._stand_sustain_counter[env_idx] = 0

    def rider_lowest_z(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.bottle.pose.q)
        half = torch.stack([self._rider_half_long, self._rider_half_z, self._rider_half_z], dim=-1)
        drop = (rot[:, 2, :].abs() * half).sum(dim=-1)
        return self.bottle.pose.p[:, 2] - drop

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        bottle_pos = self.bottle.pose.p
        bottle_vel = self.bottle.linear_velocity
        belt_v = self.belt.velocity

        long_axis_z = quaternion_to_matrix(self.bottle.pose.q)[:, 2, 0]
        is_upright = long_axis_z.abs() > math.cos(math.radians(self._upright_tol_deg))

        lowest_z = self.rider_lowest_z()
        on_deck = torch.abs(lowest_z - deck_z) <= self._deck_rest_tol

        comoving = (torch.abs(bottle_vel[:, 1] - belt_v) <= self._comove_tol) & (
            torch.abs(bottle_vel[:, 0]) <= self._lateral_speed_tol
        )

        finger_gap = self.gripper_aperture()
        released = (finger_gap >= self._rider_grasp_width + self._release_gap_margin) & (
            torch.linalg.norm(tcp_pos - bottle_pos, dim=-1) >= self._release_distance
        )

        on_belt = (torch.abs(bottle_pos[:, 0] - ox) <= self._belt_band_half) & (
            torch.abs(bottle_pos[:, 1] - oy) < self._belt_run_y
        )

        standing = is_upright & on_deck & comoving & released & on_belt
        self._stand_sustain_counter = self._advance_sustain("stand_sustain", self._stand_sustain_counter, standing)

        return {
            "success": self._stand_sustain_counter >= self._stand_sustain_steps,
            "is_upright": is_upright,
            "on_deck": on_deck,
            "comoving": comoving,
            "released": released,
            "on_belt": on_belt,
            "long_axis_z": long_axis_z.abs(),
            "lowest_z": lowest_z,
            "finger_gap": finger_gap,
            "stand_sustain": self._stand_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        bottle_pos = self.bottle.pose.p

        tcp_to_bottle = torch.linalg.norm(tcp_pos - bottle_pos, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_bottle)
        upright_reward = info["long_axis_z"].clamp(0.0, 1.0)
        stood = (info["is_upright"] & info["on_deck"] & info["comoving"] & info["on_belt"]).float()

        return reach_reward + upright_reward + 2.0 * stood + 2.0 * stood * info["released"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 6.0
