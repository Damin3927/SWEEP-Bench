from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_selector_set.assets import (
    WELL_NAMES,
    build_selector_carrier,
    build_selector_slider,
)
from sweep_bench.tasks.conveyor_selector_set.config import RandomizationConfig
from sweep_bench.tasks.conveyor_selector_set.scripted import ScriptedSelectorSetMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@bench_task("ConveyorSelectorSet-v0")
@register_env("ConveyorSelectorSet-v0", max_episode_steps=300)
class ConveyorSelectorSetEnv(ScriptedSelectorSetMixin, ConveyorWorkspaceEnv):
    """Move a passing carrier's selector to the detent position the instruction names."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _carrier_half_x: float = 0.032
    _carrier_half_y: float = 0.048
    _channel_half_x: float = 0.019
    _channel_half_y: float = 0.036
    _channel_height: float = 0.019
    _carrier_floor_half_thickness: float = 0.006
    _carrier_top_half_thickness: float = 0.003
    _slot_half_x: float = 0.013
    _carrier_density: float = 7500.0
    _carrier_friction: float = 0.05
    _carrier_patch_radius: float = 0.030

    _slider_base_half: tuple[float, float, float] = (0.0185, 0.012, 0.009)
    _slider_post_half: tuple[float, float, float] = (0.009, 0.010, 0.024)
    _slider_base_density: float = 4000.0
    _slider_post_density: float = 400.0
    _slider_friction: float = 0.05

    _well_pitch: float = 0.022
    _detent_k: float = 100.0
    _detent_damping: float = 5.3
    _detent_force_max: float = 0.8

    _well_tol: float = 0.004
    _seated_speed: float = 0.03
    _hands_off_radius: float = 0.06
    _capture_dx: float = 0.006
    _capture_dz: float = 0.004
    _changed_min: float = 0.015
    _carrier_shove_tol: float = 0.020
    _carrier_ride_v_tol: float = 0.08
    _slider_contact_radius: float = 0.09
    _selector_sustain_steps: int = 8

    default_prompt: str = "set the selector to the named position on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _carrier_origin_above_deck(self) -> float:
        return 2.0 * self._carrier_floor_half_thickness

    @property
    def _carrier_top_local_z(self) -> float:
        return self._channel_height + 2.0 * self._carrier_top_half_thickness

    @property
    def _slider_rest_local_z(self) -> float:
        return self._slider_base_half[2]

    @property
    def _well_offsets(self) -> tuple[float, ...]:
        return tuple((i - 1) * self._well_pitch for i in range(len(WELL_NAMES)))

    def _well_local_y(self, well: torch.Tensor) -> torch.Tensor:
        return (well.to(torch.float32) - 1.0) * self._well_pitch

    @property
    def episode_objects(self) -> list[str]:
        return [WELL_NAMES[int(w)] for w in self._target_well.tolist()]

    @property
    def episode_prompts(self) -> list[str]:
        return [f"set the selector to the {name} position on the conveyor belt" for name in self.episode_objects]

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        carrier_builder = build_selector_carrier(
            self.scene,
            half_x=self._carrier_half_x,
            half_y=self._carrier_half_y,
            channel_half_x=self._channel_half_x,
            channel_half_y=self._channel_half_y,
            channel_height=self._channel_height,
            floor_half_thickness=self._carrier_floor_half_thickness,
            top_half_thickness=self._carrier_top_half_thickness,
            slot_half_x=self._slot_half_x,
            density=self._carrier_density,
            friction=self._carrier_friction,
            well_offsets=self._well_offsets,
        )
        carrier_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._carrier_origin_above_deck])
        self.carrier = carrier_builder.build(name="selector_carrier")
        self.belt.register_rider(self.carrier, patch_radius=self._carrier_patch_radius)

        slider_builder = build_selector_slider(
            self.scene,
            base_half=self._slider_base_half,
            post_half=self._slider_post_half,
            base_density=self._slider_base_density,
            post_density=self._slider_post_density,
            friction=self._slider_friction,
        )
        slider_builder.initial_pose = sapien.Pose(
            p=[ox, oy, deck_z + self._carrier_origin_above_deck + self._slider_rest_local_z]
        )
        self.slider = slider_builder.build(name="selector_slider")

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.slider]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        carrier_q = self._sample_carrier_orientation(env_idx, b)
        carrier_pos = self._sample_carrier_position(env_idx, b)
        self.carrier.set_pose(Pose.create_from_pq(carrier_pos, carrier_q))
        ride_vel = torch.zeros((b, 3), device=device)
        ride_vel[:, 1] = self.belt.velocity[env_idx]
        self.carrier.set_linear_velocity(ride_vel)
        self.carrier.set_angular_velocity(torch.zeros((b, 3), device=device))

        wells = np.arange(len(WELL_NAMES))
        target = self._episode_rng.choice(wells, size=b)
        offset = self._episode_rng.choice(wells[1:], size=b)
        reset = (target + offset) % len(WELL_NAMES)
        target_t = torch.as_tensor(target.astype(np.int64), device=device)
        reset_t = torch.as_tensor(reset.astype(np.int64), device=device)

        local = torch.zeros((b, 3), device=device)
        local[:, 1] = self._well_local_y(reset_t)
        local[:, 2] = self._slider_rest_local_z
        rot = quaternion_to_matrix(carrier_q)
        slider_pos = carrier_pos + torch.bmm(rot, local.unsqueeze(-1)).squeeze(-1)
        self.slider.set_pose(Pose.create_from_pq(slider_pos, carrier_q))
        self.slider.set_linear_velocity(ride_vel.clone())
        self.slider.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_selector_trackers(
            env_idx,
            target=target_t,
            reset_s=self._well_local_y(reset_t),
            carrier_x=carrier_pos[:, 0],
            strength=self._sample_detent_strength(b),
        )

    def _sample_carrier_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.carrier_yaw:
            lo, hi = self.randomization.carrier_yaw_range
            yaw = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            yaw_t = torch.as_tensor(yaw, device=device)
            q[:, 0] = torch.cos(yaw_t * 0.5)
            q[:, 3] = torch.sin(yaw_t * 0.5)
        return q

    def _sample_detent_strength(self, b: int) -> torch.Tensor:
        device = self.device
        if not self.randomization.detent_strength:
            return torch.ones((b,), device=device)
        lo, hi = self.randomization.detent_strength_range
        return torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)

    def _sample_carrier_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._carrier_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._carrier_origin_above_deck
        return pos

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_detent_force()

    def _apply_detent_force(self) -> None:
        if not getattr(self, "_detent_ready", False):
            return
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()

        groove = self._selector_groove_axis()
        s = ((self.slider.pose.p - self.carrier.pose.p) * groove).sum(dim=-1)
        v_rel = ((self.slider.linear_velocity - self.carrier.linear_velocity) * groove).sum(dim=-1)

        nearest = torch.round(s / self._well_pitch).clamp(-1.0, 1.0)
        e = s - nearest * self._well_pitch
        k = self._detent_k * self._detent_strength_env
        f_max = self._detent_force_max * self._detent_strength_env
        f = torch.clamp(-k * e - self._detent_damping * v_rel, -f_max, f_max)
        force = groove * f.unsqueeze(-1)

        if gpu:
            force_buf = self.scene.px.cuda_rigid_body_force.torch()
            force_buf[self.slider._body_data_index, :3] = force
            self.scene.px.gpu_apply_rigid_dynamic_force()
        else:
            self.slider._bodies[0].add_force_torque(force.squeeze(0).cpu().numpy(), np.zeros(3, dtype=np.float32))

    def _selector_groove_axis(self) -> torch.Tensor:
        return quaternion_to_matrix(self.carrier.pose.q)[:, :, 1]

    def _slider_carrier_offset(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.carrier.pose.q)
        rel_world = (self.slider.pose.p - self.carrier.pose.p).unsqueeze(-1)
        return torch.bmm(rot.transpose(1, 2), rel_world).squeeze(-1)

    def _selector_target_local_y(self) -> torch.Tensor:
        return self._well_local_y(self._target_well)

    def _selector_push_sign(self) -> torch.Tensor:
        return self._selector_push_dir

    def _carrier_top_world_z(self) -> torch.Tensor:
        return self.carrier.pose.p[:, 2] + self._carrier_top_local_z

    def _carrier_half_x_extent(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.carrier.pose.q)
        half = torch.as_tensor(
            [self._carrier_half_x, self._carrier_half_y, self._carrier_top_local_z],
            device=self.device,
            dtype=rot.dtype,
        )
        return (half * rot[:, 0, :].abs()).sum(dim=-1)

    def _reset_selector_trackers(
        self,
        env_idx: torch.Tensor,
        *,
        target: torch.Tensor | None = None,
        reset_s: torch.Tensor | None = None,
        carrier_x: torch.Tensor | None = None,
        strength: torch.Tensor | None = None,
    ) -> None:
        if not hasattr(self, "_selector_sustain_counter"):
            n, device = self.num_envs, self.device
            self._selector_sustain_counter = torch.zeros(n, dtype=torch.long, device=device)
            self._target_well = torch.ones(n, dtype=torch.long, device=device)
            self._slider_reset_s = torch.zeros(n, device=device)
            self._carrier_reset_x = torch.zeros(n, device=device)
            self._detent_strength_env = torch.ones(n, device=device)
            self._selector_push_dir = torch.ones(n, device=device)
            self._slider_left_reset = torch.zeros(n, dtype=torch.bool, device=device)
            self._slider_move_was_armed = torch.zeros(n, dtype=torch.bool, device=device)
            self._slider_ever_near = torch.zeros(n, dtype=torch.bool, device=device)
        self._selector_sustain_counter[env_idx] = 0
        self._slider_left_reset[env_idx] = False
        self._slider_move_was_armed[env_idx] = False
        self._slider_ever_near[env_idx] = False
        if target is not None:
            self._target_well[env_idx] = target
        if reset_s is not None:
            self._slider_reset_s[env_idx] = reset_s
            self._selector_push_dir[env_idx] = torch.sign(self._well_local_y(self._target_well[env_idx]) - reset_s)
        if carrier_x is not None:
            self._carrier_reset_x[env_idx] = carrier_x
        if strength is not None:
            self._detent_strength_env[env_idx] = strength
        self._detent_ready = True

    def evaluate(self):
        oy = self.workspace_offset[1]
        ox = self.workspace_offset[0]
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        carrier_pos = self.carrier.pose.p
        slider_pos = self.slider.pose.p

        local = self._slider_carrier_offset()
        s = local[:, 1]
        well_error = s - self._selector_target_local_y()
        in_named_well = well_error.abs() < self._well_tol

        v_rel = self.slider.linear_velocity - self.carrier.linear_velocity
        seated = torch.linalg.norm(v_rel, dim=-1) < self._seated_speed

        tcp_to_slider = torch.linalg.norm(tcp_pos - slider_pos, dim=-1)
        hands_off = (tcp_to_slider >= self._hands_off_radius) & (tcp_pos[:, 2] > self._carrier_top_world_z())

        captured = (local[:, 0].abs() < self._capture_dx) & (
            torch.abs(local[:, 2] - self._slider_rest_local_z) < self._capture_dz
        )

        carrier_riding = torch.abs(self.carrier.linear_velocity[:, 1] - self.belt.velocity) <= self._carrier_ride_v_tol
        carrier_on_belt = torch.abs(carrier_pos[:, 1] - oy) < self._conveyor_half_length
        carrier_undisplaced = torch.abs(carrier_pos[:, 0] - self._carrier_reset_x) < self._carrier_shove_tol

        travel = (s - self._slider_reset_s).abs()
        self._slider_ever_near = self._slider_ever_near | (tcp_to_slider < self._slider_contact_radius)
        left_reset = travel > 0.5 * self._well_pitch
        first_leave = left_reset & ~self._slider_left_reset
        self._slider_move_was_armed = torch.where(first_leave, self._slider_ever_near, self._slider_move_was_armed)
        self._slider_left_reset = self._slider_left_reset | left_reset

        holding = in_named_well & seated & hands_off & captured & carrier_riding & carrier_on_belt & carrier_undisplaced
        self._selector_sustain_counter = self._advance_sustain(
            "selector_sustain", self._selector_sustain_counter, holding
        )
        success = (self._selector_sustain_counter >= self._selector_sustain_steps) & self._slider_move_was_armed

        return {
            "success": success,
            "in_named_well": in_named_well,
            "well_error": well_error,
            "seated": seated,
            "hands_off": hands_off,
            "tcp_to_slider": tcp_to_slider,
            "captured": captured,
            "carrier_riding": carrier_riding,
            "carrier_on_belt": carrier_on_belt,
            "carrier_undisplaced": carrier_undisplaced,
            "move_was_arm_caused": self._slider_move_was_armed,
            "slider_travel": travel,
            "slider_changed": travel >= self._changed_min,
            "carrier_band_margin": self._conveyor_half_width
            - self._carrier_half_x_extent()
            - torch.abs(carrier_pos[:, 0] - ox),
            "selector_sustain": self._selector_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        tcp_to_slider = info["tcp_to_slider"]

        reach_reward = 1 - torch.tanh(5 * tcp_to_slider)

        set_reward = 1.0 - (info["well_error"].abs() / self._well_pitch).clamp(0.0, 1.0)

        withdraw_reward = info["in_named_well"].to(tcp_to_slider.dtype) * (
            tcp_to_slider / self._hands_off_radius
        ).clamp(0.0, 1.0)

        intact = (info["captured"] & info["carrier_undisplaced"]).to(tcp_to_slider.dtype)
        return intact * (reach_reward + 2.0 * set_reward + withdraw_reward)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0
