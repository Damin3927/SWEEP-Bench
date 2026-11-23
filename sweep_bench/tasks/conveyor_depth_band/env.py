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

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import tolerance_speed
from sweep_bench.tasks.conveyor_depth_band.assets import (
    build_band_housing,
    build_band_plunger,
)
from sweep_bench.tasks.conveyor_depth_band.config import RandomizationConfig
from sweep_bench.tasks.conveyor_depth_band.scripted import ScriptedDepthBandMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_GRAVITY = 9.81


@bench_task("ConveyorDepthBand-v0")
@register_env("ConveyorDepthBand-v0", max_episode_steps=300)
class ConveyorDepthBandEnv(ScriptedDepthBandMixin, ConveyorWorkspaceEnv):
    """Press a plunger into its marked depth band and hold -- bottoming out is fatal."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _bore_half_width: float = 0.021
    _housing_half_width: float = 0.035
    _wall_height: float = 0.034
    _floor_half_thickness: float = 0.005

    _plunger_half_width: float = 0.019
    _plunger_half_height: float = 0.024
    _plunger_face_half_height: float = 0.004
    _plunger_rest_gap: float = 0.018

    _spring_damping: float = 6.5
    _spring_force_max: float = 12.0

    _housing_patch_radius: float = 0.03

    _band_centre_default: float = 0.0105
    _band_half_width: float = 0.0025
    _bottom_depth: float = 0.016
    _load_force_min: float = 0.2
    _band_sustain_steps: int = 7

    _housing_x_tol: float = 0.045
    _housing_y_span: float = 0.9
    _housing_tilt_tol: float = math.radians(8.0)
    _housing_comove_tol_frac: float = 0.40
    _housing_comove_tol_floor: float = 0.065
    _housing_comove_tol_ceiling: float = 0.220

    default_prompt: str = "press the plunger down to the painted band and hold it there"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        spring_k: float = 150.0,
        band_offset: float = 0.0,
        randomization: RandomizationConfig | None = None,
        reconfiguration_freq: int | None = None,
        num_envs: int = 1,
        **kwargs,
    ) -> None:
        randomization = randomization if randomization is not None else self.RandomizationConfig()
        k_lo, k_hi = randomization.spring_k_range
        if not 0.0 < k_lo <= spring_k <= k_hi:
            raise ValueError(f"spring_k {spring_k} must lie in the draw range {(k_lo, k_hi)}")
        o_lo, o_hi = randomization.band_offset_range
        if not o_lo <= band_offset <= o_hi:
            raise ValueError(f"band_offset {band_offset} must lie in the draw range {(o_lo, o_hi)}")
        if self._band_centre_default + o_hi + self._band_half_width >= self._bottom_depth:
            raise ValueError(
                f"deepest legal depth {self._band_centre_default + o_hi + self._band_half_width} must stay "
                f"under the {self._bottom_depth} m over-press latch"
            )
        if self._band_centre_default + o_lo - self._band_half_width <= 0.0:
            raise ValueError("shallowest legal depth must stay above the 0 m rest datum")
        if self._bottom_depth >= self._plunger_rest_gap:
            raise ValueError(
                f"over-press latch {self._bottom_depth} must be under the {self._plunger_rest_gap} m "
                "hard travel, or it can never fire"
            )
        self.spring_k: float = spring_k
        self.band_offset: float = band_offset
        if reconfiguration_freq is None:
            axes_on = randomization.band_offset or randomization.spring_k
            reconfiguration_freq = 1 if (num_envs == 1 and axes_on) else 0
        super().__init__(
            *args, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, randomization=randomization, **kwargs
        )

    @property
    def _plunger_rest_local_z(self) -> float:
        return self._plunger_rest_gap + self._plunger_half_height

    @property
    def _housing_origin_above_deck(self) -> float:
        return 2.0 * self._floor_half_thickness

    @property
    def _plunger_rest_top_above_deck(self) -> float:
        return self._housing_origin_above_deck + self._plunger_rest_local_z + self._plunger_half_height

    def _band_stripe_dz(self, band_centre: float) -> float:
        return self._wall_height - self._plunger_rest_local_z + band_centre

    def _draw_per_env(self, active: bool, value_range: tuple[float, float], default: float) -> np.ndarray:
        if not active:
            return np.full(self.num_envs, default, dtype=np.float32)
        lo, hi = value_range
        return self._episode_rng.uniform(lo, hi, size=self.num_envs).astype(np.float32)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        offsets = self._draw_per_env(
            self.randomization.band_offset,
            self.randomization.band_offset_range,
            self.band_offset,
        )
        stiffness = self._draw_per_env(
            self.randomization.spring_k,
            self.randomization.spring_k_range,
            self.spring_k,
        )
        self._band_centre = torch.as_tensor(
            self._band_centre_default + offsets, dtype=torch.float32, device=self.device
        )
        self._spring_k = torch.as_tensor(stiffness, dtype=torch.float32, device=self.device)

        housing_builder = build_band_housing(
            self.scene,
            bore_half_width=self._bore_half_width,
            outer_half_width=self._housing_half_width,
            wall_height=self._wall_height,
            floor_half_thickness=self._floor_half_thickness,
        )
        housing_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._housing_origin_above_deck])
        self.housing = housing_builder.build(name="band_housing")
        self.belt.register_rider(self.housing, patch_radius=self._housing_patch_radius)

        plunger_z = deck_z + self._housing_origin_above_deck + self._plunger_rest_local_z
        plungers: list[Actor] = []
        for i in range(self.num_envs):
            builder = build_band_plunger(
                self.scene,
                half_width=self._plunger_half_width,
                half_height=self._plunger_half_height,
                face_half_height=self._plunger_face_half_height,
                band_dz=self._band_stripe_dz(float(self._band_centre_default + offsets[i])),
                band_half_height=self._band_half_width,
            )
            builder.initial_pose = sapien.Pose(p=[ox, oy, plunger_z])
            builder.set_scene_idxs([i])
            plunger = builder.build(name=f"band_plunger-{i}")
            self.remove_from_state_dict_registry(plunger)
            plungers.append(plunger)
        self.plunger = Actor.merge(plungers, name="band_plunger")
        self.add_to_state_dict_registry(self.plunger)

        self._plunger_weight = float(self.plunger._bodies[0].mass) * _GRAVITY

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        q = self._sample_housing_orientation(env_idx, b)

        housing_pos = self._sample_housing_position(env_idx, b)
        self.housing.set_pose(Pose.create_from_pq(housing_pos, q))
        housing_vel = torch.zeros((b, 3), device=device)
        housing_vel[:, 1] = self.belt.velocity[env_idx]
        self.housing.set_linear_velocity(housing_vel)
        self.housing.set_angular_velocity(torch.zeros((b, 3), device=device))

        plunger_pos = housing_pos.clone()
        plunger_pos[:, 2] = housing_pos[:, 2] + self._plunger_rest_local_z
        self.plunger.set_pose(Pose.create_from_pq(plunger_pos, q))
        plunger_vel = torch.zeros((b, 3), device=device)
        plunger_vel[:, 1] = self.belt.velocity[env_idx]
        self.plunger.set_linear_velocity(plunger_vel)
        self.plunger.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_band_trackers(env_idx)
        self._spring_ready = True

    def _sample_housing_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.housing_yaw:
            yaw = self._episode_rng.uniform(-np.pi / 4, np.pi / 4, size=b).astype(np.float32)
            yaw_t = torch.as_tensor(yaw, device=device)
            q[:, 0] = torch.cos(yaw_t * 0.5)
            q[:, 3] = torch.sin(yaw_t * 0.5)
        return q

    def _sample_housing_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._housing_half_width - 0.04
        housing_pos = torch.zeros((b, 3), device=device)
        housing_pos[:, 0] = ox
        housing_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        housing_pos[:, 2] = deck_z + self._housing_origin_above_deck
        return housing_pos

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_plunger_force()

    def _apply_plunger_force(self) -> None:
        if not getattr(self, "_spring_ready", False):
            return
        device = self.device
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()
        anchor_z = self.housing.pose.p[:, 2] + self._plunger_rest_local_z
        disp = anchor_z - self.plunger.pose.p[:, 2]
        vz = self.plunger.linear_velocity[:, 2]
        fz = (self._spring_k * disp + self._plunger_weight - self._spring_damping * vz).clamp(
            0.0, self._spring_force_max
        )
        force = torch.zeros((self.num_envs, 3), device=device)
        force[:, 2] = fz
        if gpu:
            force_buf = self.scene.px.cuda_rigid_body_force.torch()
            force_buf[self.plunger._body_data_index, :3] = force
            self.scene.px.gpu_apply_rigid_dynamic_force()
        else:
            self.plunger._bodies[0].add_force_torque(
                force.squeeze(0).cpu().numpy().astype(np.float32), np.zeros(3, dtype=np.float32)
            )

    def _plunger_in_housing_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        rot = quaternion_to_matrix(self.housing.pose.q)
        rel_world = self.plunger.pose.p - self.housing.pose.p
        rel = torch.einsum("bji,bj->bi", rot, rel_world)
        return rel, rot

    def plunger_depth(self) -> torch.Tensor:
        rel, _rot = self._plunger_in_housing_frame()
        return self._plunger_rest_local_z - rel[:, 2]

    def _plunger_finger_force(self) -> torch.Tensor:
        agent = cast("Panda", self.agent)
        f1 = torch.linalg.norm(self.scene.get_pairwise_contact_forces(agent.finger1_link, self.plunger), dim=-1)
        f2 = torch.linalg.norm(self.scene.get_pairwise_contact_forces(agent.finger2_link, self.plunger), dim=-1)
        return torch.maximum(f1, f2)

    def _reset_band_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_band_sustain_counter"):
            self._band_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._overpressed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._band_sustain_counter[env_idx] = 0
        self._overpressed[env_idx] = False

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        rel, rot = self._plunger_in_housing_frame()
        depth = self._plunger_rest_local_z - rel[:, 2]

        band_error = depth - self._band_centre
        in_band = torch.abs(band_error) <= self._band_half_width

        finger_force = self._plunger_finger_force()
        loaded = finger_force > self._load_force_min

        housing_pos = self.housing.pose.p
        comove_tol = torch.clamp(
            self._housing_comove_tol_frac * tolerance_speed(self.belt.velocity),
            min=self._housing_comove_tol_floor,
            max=self._housing_comove_tol_ceiling,
        )
        level = rot[:, 2, 2] > math.cos(self._housing_tilt_tol)
        housing_on_belt = (
            (housing_pos[:, 2] > deck_z)
            & (torch.abs(housing_pos[:, 0] - ox) <= self._housing_x_tol)
            & (torch.abs(housing_pos[:, 1] - oy) <= self._housing_y_span)
            & level
            & (torch.abs(self.housing.linear_velocity[:, 1] - self.belt.velocity) <= comove_tol)
        )

        self._overpressed = self._overpressed | (depth > self._bottom_depth)

        holding = in_band & loaded & housing_on_belt
        self._band_sustain_counter = self._advance_sustain("depth_band", self._band_sustain_counter, holding)
        success = (self._band_sustain_counter >= self._band_sustain_steps) & ~self._overpressed

        return {
            "success": success,
            "in_band": in_band,
            "loaded": loaded,
            "housing_on_belt": housing_on_belt,
            "overpressed": self._overpressed,
            "plunger_depth": depth,
            "band_centre": self._band_centre,
            "band_error": band_error,
            "finger_force": finger_force,
            "band_sustain": self._band_sustain_counter,
            "plunger_bore_offset": torch.linalg.norm(rel[:, :2], dim=-1),
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p

        tcp_to_plunger = torch.linalg.norm(tcp_pos - self.plunger.pose.p, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_plunger)

        band_reward = 1 - torch.tanh(60.0 * torch.abs(info["band_error"]))
        alive = (~info["overpressed"]).to(torch.float32)

        return reach_reward + 2.0 * band_reward * alive + 2.0 * info["success"].to(torch.float32)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
