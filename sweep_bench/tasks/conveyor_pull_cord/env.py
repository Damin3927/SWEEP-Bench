from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_at_rest_height, rider_is_level, tolerance_speed
from sweep_bench.tasks.conveyor_pull_cord.assets import build_cord_body, build_pull_handle
from sweep_bench.tasks.conveyor_pull_cord.config import RandomizationConfig
from sweep_bench.tasks.conveyor_pull_cord.scripted import ScriptedPullCordMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@bench_task("ConveyorPullCord-v0")
@register_env("ConveyorPullCord-v0", max_episode_steps=300)
class ConveyorPullCordEnv(ScriptedPullCordMixin, ConveyorWorkspaceEnv):
    """Pull a spring-loaded starter handle through its stroke on a passing rider."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    conveyor_speed: float = 0.10

    _base_half_xy: float = 0.040
    _base_half_thickness: float = 0.006
    _tube_outer_half: float = 0.024
    _bore_half: float = 0.014
    _tube_height: float = 0.082
    _lip_opening_half: float = 0.0075
    _lip_thickness: float = 0.006
    _body_density: float = 2200.0
    _body_patch_radius: float = 0.030

    _collar_half: float = 0.011
    _collar_half_thickness: float = 0.006
    _shaft_half: float = 0.005
    _bar_half_x: float = 0.006
    _bar_half_y: float = 0.012
    _bar_half_z: float = 0.006
    _handle_rest_local_z: float = 0.118
    _handle_density: float = 2000.0

    _handle_spring_k: float = 25.0
    _handle_spring_damping: float = 2.0
    _handle_force_max: float = 4.0

    _pull_success_travel: float = 0.060
    _pull_onset_travel: float = 0.005
    _grip_gap_tol: float = 0.003
    _grip_force_threshold: float = 0.05
    _release_finger_width: float = 0.035
    _return_home_travel: float = 0.010
    _return_sustain_steps: int = 4
    _body_lift_tol: float = 0.006
    _ride_slip_abs: float = 0.080
    _ride_slip_rel: float = 0.30
    _belt_half_width: float = 0.08

    default_prompt: str = "pull the starter handle on the conveyor and let it snap back"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def hard_travel(self) -> float:
        return self._tube_height - 2.0 * self._collar_half_thickness

    @property
    def _body_origin_above_deck(self) -> float:
        return 2.0 * self._base_half_thickness

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        body_builder = build_cord_body(
            self.scene,
            base_half_xy=self._base_half_xy,
            base_half_thickness=self._base_half_thickness,
            tube_outer_half=self._tube_outer_half,
            bore_half=self._bore_half,
            tube_height=self._tube_height,
            lip_opening_half=self._lip_opening_half,
            lip_thickness=self._lip_thickness,
            density=self._body_density,
        )
        body_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._body_origin_above_deck])
        self.body = body_builder.build(name="cord_body")
        self.belt.register_rider(self.body, patch_radius=self._body_patch_radius)

        handle_builder = build_pull_handle(
            self.scene,
            collar_half=self._collar_half,
            collar_half_thickness=self._collar_half_thickness,
            shaft_half=self._shaft_half,
            bar_half_x=self._bar_half_x,
            bar_half_y=self._bar_half_y,
            bar_half_z=self._bar_half_z,
            rest_local_z=self._handle_rest_local_z,
            density=self._handle_density,
        )
        handle_builder.initial_pose = sapien.Pose(
            p=[ox, oy, deck_z + self._body_origin_above_deck + self._handle_rest_local_z]
        )
        self.handle = handle_builder.build(name="pull_handle")

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.handle]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        q = self._sample_body_orientation(env_idx, b)
        body_pos = self._sample_body_position(env_idx, b)
        self.body.set_pose(Pose.create_from_pq(body_pos, q))

        handle_pos = body_pos.clone()
        handle_pos[:, 2] = body_pos[:, 2] + self._handle_rest_local_z
        self.handle.set_pose(Pose.create_from_pq(handle_pos, q))

        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        zeros = torch.zeros((b, 3), device=device)
        for actor in (self.body, self.handle):
            actor.set_linear_velocity(vel)
            actor.set_angular_velocity(zeros)

        self._sample_spring_stiffness(env_idx, b)
        self._reset_pull_cord_trackers(env_idx)
        self._handle_spring_ready = True

    def _sample_body_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.rider_yaw:
            lo, hi = self.randomization.yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
            q[:, 0] = torch.cos(0.5 * yaw)
            q[:, 3] = torch.sin(0.5 * yaw)
        return q

    def _sample_body_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._base_half_xy - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.rider_lateral_offset:
            lo, hi = self.randomization.lateral_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = pos[:, 0] + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._body_origin_above_deck
        return pos

    def _sample_spring_stiffness(self, env_idx: torch.Tensor, b: int) -> None:
        if not hasattr(self, "_handle_spring_k_env"):
            self._handle_spring_k_env = torch.full((self.num_envs,), self._handle_spring_k, device=self.device)
        if self.randomization.spring_stiffness:
            lo, hi = self.randomization.spring_k_range
            frac = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            self._handle_spring_k_env[env_idx] = self._handle_spring_k * torch.as_tensor(frac, device=self.device)
        else:
            self._handle_spring_k_env[env_idx] = self._handle_spring_k

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_handle_return_force()

    def _apply_handle_return_force(self) -> None:
        if not getattr(self, "_handle_spring_ready", False):
            return
        device = self.device
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()
        anchor_z = self.body.pose.p[:, 2] + self._handle_rest_local_z
        disp = (self.handle.pose.p[:, 2] - anchor_z).clamp(min=0.0)
        vz = self.handle.linear_velocity[:, 2]
        fz = (-(self._handle_spring_k_env * disp + self._handle_spring_damping * vz)).clamp(
            -self._handle_force_max, 0.0
        )
        force = torch.zeros((self.num_envs, 3), device=device)
        force[:, 2] = fz
        if gpu:
            force_buf = self.scene.px.cuda_rigid_body_force.torch()
            force_buf[self.handle._body_data_index, :3] = force
            self.scene.px.gpu_apply_rigid_dynamic_force()
        else:
            self.handle._bodies[0].add_force_torque(force.squeeze(0).cpu().numpy(), np.zeros(3, dtype=np.float32))

    def handle_travel(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.body.pose.q)
        rel_world = self.handle.pose.p - self.body.pose.p
        rel_z = torch.einsum("bj,bj->b", rot[:, :, 2], rel_world)
        return rel_z - self._handle_rest_local_z

    def travel_latched(self) -> torch.Tensor:
        if not hasattr(self, "_travel_latch"):
            self._travel_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._travel_latch

    def _finger_separation(self) -> torch.Tensor:
        return self.gripper_aperture()

    def _handle_is_gripped(self) -> torch.Tensor:
        agent = cast("Panda", self.agent)
        f1 = torch.linalg.norm(self.scene.get_pairwise_contact_forces(agent.finger1_link, self.handle), dim=-1)
        f2 = torch.linalg.norm(self.scene.get_pairwise_contact_forces(agent.finger2_link, self.handle), dim=-1)
        pinched = (f1 > self._grip_force_threshold) & (f2 > self._grip_force_threshold)
        on_the_bar = torch.abs(self._finger_separation() - 2.0 * self._bar_half_x) <= self._grip_gap_tol
        return pinched & on_the_bar

    def _body_on_belt(self) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        rot = quaternion_to_matrix(self.body.pose.q)
        half = torch.as_tensor(
            [self._base_half_xy, self._base_half_xy, self._base_half_thickness],
            device=self.device,
            dtype=rot.dtype,
        )
        half_x_extent = (half * rot[:, 0, :].abs()).sum(dim=-1)
        pos = self.body.pose.p
        return (
            (torch.abs(pos[:, 0] - ox) + half_x_extent < self._belt_half_width)
            & (torch.abs(pos[:, 1] - oy) < self._conveyor_half_length)
            & rider_at_rest_height(
                self.body.pose,
                deck_z=deck_z,
                rest_height=self._body_origin_above_deck,
                max_rise=self._body_lift_tol,
            )
            & rider_is_level(self.body.pose)
        )

    def _reset_pull_cord_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_return_counter"):
            self._return_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._pull_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._ride_fault = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._travel_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._return_counter[env_idx] = 0
        self._pull_started[env_idx] = False
        self._ride_fault[env_idx] = False
        self._travel_latch[env_idx] = False

    def evaluate(self):
        oz = self.workspace_offset[2]
        deck_z = oz + self._conveyor_deck_height

        travel = self.handle_travel()
        on_belt = self._body_on_belt()

        resting_z = deck_z + self._body_origin_above_deck
        seated = torch.abs(self.body.pose.p[:, 2] - resting_z) <= self._body_lift_tol
        belt_v = self.belt.velocity
        slip_tol = torch.clamp(tolerance_speed(belt_v) * self._ride_slip_rel, min=self._ride_slip_abs)
        riding = torch.abs(self.body.linear_velocity[:, 1] - belt_v) <= slip_tol
        self._pull_started = self._pull_started | (travel > self._pull_onset_travel)
        self._ride_fault = self._ride_fault | (self._pull_started & ~(seated & riding))

        pulled = travel >= self._pull_success_travel
        gripped = self._handle_is_gripped()
        self._travel_latch = self._travel_latch | (pulled & gripped & on_belt & ~self._ride_fault)

        released = self._finger_separation() >= self._release_finger_width
        home = travel <= self._return_home_travel
        returning = self._travel_latch & released & home & on_belt
        self._return_counter = self._advance_sustain("pull_cord_return", self._return_counter, returning)
        success = self._return_counter >= self._return_sustain_steps

        return {
            "success": success,
            "handle_travel": travel,
            "handle_pulled": pulled,
            "handle_gripped": gripped,
            "body_on_belt": on_belt,
            "body_seated": seated,
            "body_riding": riding,
            "pull_started": self._pull_started,
            "ride_fault": self._ride_fault,
            "travel_latched": self._travel_latch,
            "gripper_released": released,
            "handle_home": home,
            "return_sustain": self._return_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p

        tcp_to_bar = torch.linalg.norm(tcp_pos - self.handle.pose.p, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_bar)

        stroke_reward = (info["handle_travel"] / self._pull_success_travel).clamp(0.0, 1.0)
        latched_reward = info["travel_latched"].to(torch.float32)

        return reach_reward + 2.0 * stroke_reward + latched_reward + 2.0 * info["success"].to(torch.float32)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 6.0
