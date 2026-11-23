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
from sweep_bench.tasks.conveyor_spring_hold.assets import (
    build_spring_tab,
    build_spring_tab_carrier,
)
from sweep_bench.tasks.conveyor_spring_hold.config import RandomizationConfig
from sweep_bench.tasks.conveyor_spring_hold.scripted import ScriptedSpringHoldMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorSpringHold-v0")
@register_env("ConveyorSpringHold-v0", max_episode_steps=300)
class ConveyorSpringHoldEnv(ScriptedSpringHoldMixin, ConveyorWorkspaceEnv):
    """Hold a spring-returned tab at the marked extension while it rides the belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _carrier_half_x: float = 0.050
    _carrier_half_y: float = 0.048
    _plate_half_z: float = 0.005
    _rail_inner_y: float = 0.020
    _rail_outer_y: float = 0.030
    _rail_height: float = 0.032
    _stop_thickness: float = 0.006
    _stop_height: float = 0.012
    _band_half_x: float = 0.002
    _carrier_density: float = 4800.0
    _carrier_patch_radius: float = 0.030

    _tab_half_x: float = 0.018
    _tab_half_y: float = 0.018
    _tab_half_z: float = 0.036
    _tab_density: float = 500.0
    _tab_friction: float = 0.06
    _tab_rest_dx: float = 0.020
    _tab_hard_travel: float = 0.046

    _spring_damping: float = 2.2
    _spring_force_max: float = 6.0

    _hold_band: float = 0.005
    _grip_squeeze_max: float = 0.007
    _grip_slack_max: float = 0.003
    _tab_capture_dz: float = 0.006
    _tab_capture_lateral: float = 0.008
    _bottom_out_extension: float = 0.040
    _hold_sustain_steps: int = 6

    _carrier_x_tol: float = 0.030
    _carrier_y_span: float = 0.85
    _carrier_tilt_tol: float = math.radians(10.0)
    _carrier_comove_tol_frac: float = 0.25
    _carrier_comove_tol_floor: float = 0.065
    _carrier_comove_tol_ceiling: float = 0.110

    default_prompt: str = "pull the tab out to the marked band and hold it there"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        target_extension: float = 0.025,
        spring_stiffness: float = 25.0,
        randomization: RandomizationConfig | None = None,
        reconfiguration_freq: int | None = None,
        num_envs: int = 1,
        **kwargs,
    ) -> None:
        randomization = randomization if randomization is not None else self.RandomizationConfig()
        lo, hi = randomization.target_extension_range
        if not 0.0 < lo <= target_extension <= hi:
            raise ValueError(f"target_extension {target_extension} must lie in the draw range {(lo, hi)}")
        if hi + self._hold_band >= self._bottom_out_extension:
            raise ValueError(
                f"deepest target {hi} + band {self._hold_band} must stay under the "
                f"{self._bottom_out_extension} m bottom-out kill threshold"
            )
        if self._bottom_out_extension >= self._tab_hard_travel:
            raise ValueError(
                f"kill threshold {self._bottom_out_extension} must be under the "
                f"{self._tab_hard_travel} m hard travel, or it can never fire"
            )
        k_lo, k_hi = randomization.spring_stiffness_range
        if not 0.0 < k_lo <= spring_stiffness <= k_hi:
            raise ValueError(f"spring_stiffness {spring_stiffness} must lie in the draw range {(k_lo, k_hi)}")
        self.target_extension: float = target_extension
        self.spring_stiffness: float = spring_stiffness
        if reconfiguration_freq is None:
            axes_on = randomization.target_extension or randomization.spring_stiffness
            reconfiguration_freq = 1 if (num_envs == 1 and axes_on) else 0
        super().__init__(
            *args, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, randomization=randomization, **kwargs
        )

    @property
    def _tab_rest_dz(self) -> float:
        return self._plate_half_z + self._tab_half_z

    @property
    def _tab_grasp_width(self) -> float:
        return 2.0 * self._tab_half_x

    @property
    def _carrier_origin_above_deck(self) -> float:
        return self._plate_half_z

    _tab_grasp_below_top: float = 0.022

    @property
    def _tab_grasp_z_above(self) -> float:
        return 2.0 * self._plate_half_z + 2.0 * self._tab_half_z - self._tab_grasp_below_top

    @property
    def _stop_inner_x(self) -> float:
        return self._tab_rest_dx - self._tab_hard_travel - self._tab_half_x

    def _draw_per_env(self, active: bool, value_range: tuple[float, float], default: float) -> np.ndarray:
        if not active:
            return np.full(self.num_envs, default, dtype=np.float32)
        lo, hi = value_range
        return self._episode_rng.uniform(lo, hi, size=self.num_envs).astype(np.float32)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        targets = self._draw_per_env(
            self.randomization.target_extension,
            self.randomization.target_extension_range,
            self.target_extension,
        )
        stiffness = self._draw_per_env(
            self.randomization.spring_stiffness,
            self.randomization.spring_stiffness_range,
            self.spring_stiffness,
        )
        self._target_extension = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        self._spring_stiffness = torch.as_tensor(stiffness, dtype=torch.float32, device=self.device)

        carriers: list[Actor] = []
        for i in range(self.num_envs):
            builder = build_spring_tab_carrier(
                self.scene,
                half_x=self._carrier_half_x,
                half_y=self._carrier_half_y,
                plate_half_z=self._plate_half_z,
                rail_inner_y=self._rail_inner_y,
                rail_outer_y=self._rail_outer_y,
                rail_height=self._rail_height,
                stop_inner_x=self._stop_inner_x,
                stop_thickness=self._stop_thickness,
                stop_height=self._stop_height,
                band_x=self._tab_rest_dx - float(targets[i]) - self._tab_half_x,
                band_half_x=self._band_half_x,
                density=self._carrier_density,
            )
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._carrier_origin_above_deck])
            builder.set_scene_idxs([i])
            carrier = builder.build(name=f"spring_carrier-{i}")
            self.remove_from_state_dict_registry(carrier)
            carriers.append(carrier)

        self.carrier = Actor.merge(carriers, name="spring_carrier")
        self.add_to_state_dict_registry(self.carrier)
        self.belt.register_rider(self.carrier, patch_radius=self._carrier_patch_radius)

        tab_builder = build_spring_tab(
            self.scene,
            half_x=self._tab_half_x,
            half_y=self._tab_half_y,
            half_z=self._tab_half_z,
            density=self._tab_density,
            friction=self._tab_friction,
        )
        tab_builder.initial_pose = sapien.Pose(
            p=[ox + self._tab_rest_dx, oy, deck_z + self._tab_rest_dz],
        )
        self.tab = tab_builder.build(name="spring_tab")

    def _after_reconfigure(self, options: dict) -> None:
        super()._after_reconfigure(options)
        com = np.stack([np.asarray(body.cmass_local_pose.p) for body in self.carrier._bodies])
        self._carrier_com_local = torch.as_tensor(com, dtype=torch.float32, device=self.device)

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.tab]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        carrier_xy = self._sample_carrier_xy(env_idx, b)
        yaw = self._sample_carrier_yaw(env_idx, b)
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = torch.cos(0.5 * yaw)
        q[:, 3] = torch.sin(0.5 * yaw)

        carrier_pos = torch.zeros((b, 3), device=device)
        carrier_pos[:, 0] = carrier_xy[:, 0]
        carrier_pos[:, 1] = carrier_xy[:, 1]
        carrier_pos[:, 2] = deck_z + self._carrier_origin_above_deck
        self.carrier.set_pose(Pose.create_from_pq(carrier_pos, q))

        tab_pos = torch.zeros((b, 3), device=device)
        tab_pos[:, 0] = carrier_pos[:, 0] + self._tab_rest_dx * cos
        tab_pos[:, 1] = carrier_pos[:, 1] + self._tab_rest_dx * sin
        tab_pos[:, 2] = deck_z + self._tab_rest_dz
        self.tab.set_pose(Pose.create_from_pq(tab_pos, q))

        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        zeros = torch.zeros((b, 3), device=device)
        for actor in (self.carrier, self.tab):
            actor.set_linear_velocity(vel)
            actor.set_angular_velocity(zeros)

        self._reset_hold_trackers(env_idx)
        self._spring_ready = True

    def _sample_carrier_xy(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        device = self.device
        start_y = self._conveyor_half_length - self._carrier_half_y - 0.04
        xy = torch.zeros((b, 2), device=device)
        xy[:, 0] = ox
        xy[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        return xy

    def _sample_carrier_yaw(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        if not self.randomization.carrier_yaw:
            return torch.zeros((b,), device=self.device)
        lo, hi = self.randomization.carrier_yaw_range
        yaw = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        return torch.as_tensor(yaw, device=self.device)

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_tab_spring_force()

    def _apply_tab_spring_force(self) -> None:
        if not getattr(self, "_spring_ready", False):
            return
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()
        rot = quaternion_to_matrix(self.carrier.pose.q)
        u = rot[:, :, 0]
        rel = self.tab.pose.p - self.carrier.pose.p
        x_local = torch.einsum("bj,bj->b", u, rel)
        extension = self._tab_rest_dx - x_local
        v_rel = self.tab.linear_velocity - self.carrier.linear_velocity
        v_along = torch.einsum("bj,bj->b", u, v_rel)
        f_mag = (self._spring_stiffness * extension - self._spring_damping * v_along).clamp(
            -self._spring_force_max, self._spring_force_max
        )
        force = u * f_mag.unsqueeze(-1)
        reaction = -force
        com_local = self._carrier_com_local.expand(self.num_envs, 3)
        lever = self.tab.pose.p - (self.carrier.pose * Pose.create_from_pq(com_local)).p
        reaction_torque = torch.linalg.cross(lever, reaction, dim=-1)
        if gpu:
            force_buf = self.scene.px.cuda_rigid_body_force.torch()
            torque_buf = self.scene.px.cuda_rigid_body_torque.torch()
            force_buf[self.tab._body_data_index, :3] = force
            force_buf[self.carrier._body_data_index, :3] += reaction
            torque_buf[self.carrier._body_data_index, :3] += reaction_torque
            self.scene.px.gpu_apply_rigid_dynamic_force()
            self.scene.px.gpu_apply_rigid_dynamic_torque()
        else:
            self.tab._bodies[0].add_force_torque(
                force.squeeze(0).cpu().numpy().astype(np.float32), np.zeros(3, dtype=np.float32)
            )
            self.carrier._bodies[0].add_force_torque(
                reaction.squeeze(0).cpu().numpy().astype(np.float32),
                reaction_torque.squeeze(0).cpu().numpy().astype(np.float32),
            )

    def _tab_in_carrier_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        rot = quaternion_to_matrix(self.carrier.pose.q)
        rel_world = self.tab.pose.p - self.carrier.pose.p
        rel = torch.einsum("bji,bj->bi", rot, rel_world)
        return rel, rot

    def tab_extension(self) -> torch.Tensor:
        rel, _rot = self._tab_in_carrier_frame()
        return self._tab_rest_dx - rel[:, 0]

    def _reset_hold_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_hold_sustain_counter"):
            self._hold_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._bottom_out_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold_sustain_counter[env_idx] = 0
        self._bottom_out_latch[env_idx] = False

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        rel, rot = self._tab_in_carrier_frame()

        extension = self._tab_rest_dx - rel[:, 0]
        extension_error = extension - self._target_extension
        at_mark = torch.abs(extension_error) <= self._hold_band

        agent = cast("Panda", self.agent)
        finger_gap = self.gripper_aperture()
        width = self._tab_grasp_width
        gripped = (
            (finger_gap > width - self._grip_squeeze_max)
            & (finger_gap < width + self._grip_slack_max)
            & agent.is_grasping(self.tab)
        )

        seated = torch.abs(rel[:, 2] - self._tab_rest_dz) < self._tab_capture_dz
        centred = torch.abs(rel[:, 1]) < self._tab_capture_lateral
        captured = seated & centred

        self._bottom_out_latch = self._bottom_out_latch | (extension > self._bottom_out_extension)

        carrier_pos = self.carrier.pose.p
        comove_tol = torch.clamp(
            self._carrier_comove_tol_frac * tolerance_speed(self.belt.velocity),
            min=self._carrier_comove_tol_floor,
            max=self._carrier_comove_tol_ceiling,
        )
        level = rot[:, 2, 2] > math.cos(self._carrier_tilt_tol)
        carrier_on_belt = (
            (carrier_pos[:, 2] > deck_z)
            & (torch.abs(carrier_pos[:, 0] - ox) <= self._carrier_x_tol)
            & (torch.abs(carrier_pos[:, 1] - oy) <= self._carrier_y_span)
            & level
            & (torch.abs(self.carrier.linear_velocity[:, 1] - self.belt.velocity) <= comove_tol)
        )

        holding = at_mark & gripped & captured & carrier_on_belt
        self._hold_sustain_counter = self._advance_sustain("spring_hold", self._hold_sustain_counter, holding)
        success = (self._hold_sustain_counter >= self._hold_sustain_steps) & ~self._bottom_out_latch

        return {
            "success": success,
            "at_mark": at_mark,
            "gripped": gripped,
            "captured": captured,
            "tab_seated": seated,
            "tab_centred": centred,
            "bottomed_out": self._bottom_out_latch,
            "carrier_on_belt": carrier_on_belt,
            "tab_extension": extension,
            "target_extension": self._target_extension,
            "extension_error": extension_error,
            "finger_gap": finger_gap,
            "hold_sustain": self._hold_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p

        tcp_to_tab = torch.linalg.norm(tcp_pos - self.tab.pose.p, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_tab)

        mark_reward = 1 - torch.tanh(20.0 * torch.abs(info["extension_error"]))
        alive = (~info["bottomed_out"]).to(torch.float32)

        return reach_reward + 2.0 * mark_reward * alive + 2.0 * info["success"].to(torch.float32)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
