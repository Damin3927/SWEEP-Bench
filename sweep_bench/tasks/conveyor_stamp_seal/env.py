from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import carried_speed
from sweep_bench.tasks.conveyor_stamp_seal.assets import (
    build_seal_pad,
    build_seal_plunger,
    build_stamp,
)
from sweep_bench.tasks.conveyor_stamp_seal.config import RandomizationConfig
from sweep_bench.tasks.conveyor_stamp_seal.scripted import ScriptedStampSealMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@bench_task("ConveyorStampSeal-v0")
@register_env("ConveyorStampSeal-v0", max_episode_steps=300)
class ConveyorStampSealEnv(ScriptedStampSealMixin, ConveyorWorkspaceEnv):
    """Stamp a seal pad riding the conveyor belt with a HELD tool, then lift off."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _bore_half_width: float = 0.038
    _pad_half_width: float = 0.048
    _collar_height: float = 0.030
    _floor_half_thickness: float = 0.005

    _plunger_half_width: float = 0.036
    _plunger_half_height: float = 0.006
    _plunger_rest_gap: float = 0.010

    _detent_spring_k: float = 400.0
    _detent_damping: float = 12.0
    _detent_hold_force: float = 2.0
    _detent_force_max: float = 25.0
    _detent_depth: float = 0.005

    _stamp_shaft_radius: float = 0.009
    _stamp_shaft_above_grip: float = 0.015
    _stamp_shaft_below_grip: float = 0.045
    _stamp_head_radius: float = 0.018
    _stamp_head_half_thickness: float = 0.006

    _pad_patch_radius: float = 0.03

    _pad_spawn_max_dist: float = 0.40

    _latch_success_depth: float = 0.007
    _liftoff_tcp_clearance: float = 0.11
    _stamp_held_distance: float = 0.03
    _pad_upright_cos: float = math.cos(math.radians(10.0))
    _pad_travel_frac: float = 0.5
    _seal_sustain_steps: int = 8

    default_prompt: str = "stamp the seal pad on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _pad_origin_above_deck(self) -> float:
        return 2.0 * self._floor_half_thickness

    @property
    def _plunger_rest_local_z(self) -> float:
        return self._plunger_rest_gap + self._plunger_half_height

    @property
    def _plunger_pressed_top_local_z(self) -> float:
        return 2.0 * self._plunger_half_height

    @property
    def _stamp_face_below_tcp(self) -> float:
        return self._stamp_shaft_below_grip + 2.0 * self._stamp_head_half_thickness

    @property
    def _stamp_clear_half_width(self) -> float:
        return self._bore_half_width - self._stamp_head_radius

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        pad_builder = build_seal_pad(
            self.scene,
            bore_half_width=self._bore_half_width,
            outer_half_width=self._pad_half_width,
            collar_height=self._collar_height,
            floor_half_thickness=self._floor_half_thickness,
        )
        pad_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._pad_origin_above_deck])
        self.pad = pad_builder.build(name="seal_pad")
        self.belt.register_rider(self.pad, patch_radius=self._pad_patch_radius)

        plunger_builder = build_seal_plunger(
            self.scene,
            half_width=self._plunger_half_width,
            half_height=self._plunger_half_height,
            target_radius=self._stamp_head_radius,
        )
        plunger_builder.initial_pose = sapien.Pose(
            p=[ox, oy, deck_z + self._pad_origin_above_deck + self._plunger_rest_local_z]
        )
        self.plunger = plunger_builder.build(name="seal_plunger")

        stamp_builder = build_stamp(
            self.scene,
            shaft_radius=self._stamp_shaft_radius,
            shaft_above_grip=self._stamp_shaft_above_grip,
            shaft_below_grip=self._stamp_shaft_below_grip,
            head_radius=self._stamp_head_radius,
            head_half_thickness=self._stamp_head_half_thickness,
        )
        stamp_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.stamp = stamp_builder.build(name="stamp")

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = int(env_idx.shape[0])
        return 2.0 * np.full((b, 1), self._stamp_shaft_radius - 0.0005, dtype=np.float32)

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.plunger, self.stamp]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        pad_q = self._sample_pad_orientation(env_idx, b)

        pad_pos = self._sample_pad_position(env_idx, b)
        self.pad.set_pose(Pose.create_from_pq(pad_pos, pad_q))
        pad_vel = torch.zeros((b, 3), device=device)
        pad_vel[:, 1] = self.belt.velocity[env_idx]
        self.pad.set_linear_velocity(pad_vel)
        self.pad.set_angular_velocity(torch.zeros((b, 3), device=device))

        plunger_pos = pad_pos.clone()
        plunger_pos[:, 2] = pad_pos[:, 2] + self._plunger_rest_local_z
        self.plunger.set_pose(Pose.create_from_pq(plunger_pos, pad_q))
        plunger_vel = torch.zeros((b, 3), device=device)
        plunger_vel[:, 1] = self.belt.velocity[env_idx]
        self.plunger.set_linear_velocity(plunger_vel)
        self.plunger.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pos = cast("Panda", self.agent).tcp.pose.p[env_idx]
        stamp_q = torch.zeros((b, 4), device=device)
        stamp_q[:, 0] = math.sqrt(0.5)
        stamp_q[:, 2] = math.sqrt(0.5)
        self.stamp.set_pose(Pose.create_from_pq(tcp_pos, stamp_q))
        self.stamp.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.stamp.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_seal_trackers(env_idx, b)
        self._detent_ready = True

    def _sample_pad_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.pad_yaw:
            lo, hi = self.randomization.pad_yaw_range
            yaw = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            yaw_t = torch.as_tensor(yaw, device=device)
            q[:, 0] = torch.cos(yaw_t * 0.5)
            q[:, 3] = torch.sin(yaw_t * 0.5)
        return q

    def _sample_detent_scale(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        if not self.randomization.detent_stiffness:
            return torch.ones(b, device=self.device)
        lo, hi = self.randomization.detent_stiffness_range
        draw = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        return torch.as_tensor(draw, device=self.device)

    def _sample_pad_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = min(
            self._conveyor_half_length - self._pad_half_width - 0.04,
            self._pad_spawn_max_dist,
        )
        pad_pos = torch.zeros((b, 3), device=device)
        pad_pos[:, 0] = ox
        pad_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pad_pos[:, 2] = deck_z + self._pad_origin_above_deck
        return pad_pos

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_plunger_detent_force()

    def _apply_plunger_detent_force(self) -> None:
        if not getattr(self, "_detent_ready", False):
            return
        device = self.device
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()
        depth = self._plunger_depth()
        vz = self.plunger.linear_velocity[:, 2]
        spring_up = (self._detent_spring_k * self._detent_scale * depth - self._detent_damping * vz).clamp(
            0.0, self._detent_force_max
        )
        hold_down = (-self._detent_hold_force - self._detent_damping * vz).clamp(-self._detent_force_max, 0.0)
        fz = torch.where(depth >= self._detent_depth, hold_down, spring_up)
        force = torch.zeros((self.num_envs, 3), device=device)
        force[:, 2] = fz
        if gpu:
            force_buf = self.scene.px.cuda_rigid_body_force.torch()
            force_buf[self.plunger._body_data_index, :3] = force
            self.scene.px.gpu_apply_rigid_dynamic_force()
        else:
            self.plunger._bodies[0].add_force_torque(force.squeeze(0).cpu().numpy(), np.zeros(3, dtype=np.float32))

    def _plunger_depth(self) -> torch.Tensor:
        rel = self.plunger.pose.p - self.pad.pose.p
        pad_up = quaternion_to_matrix(self.pad.pose.q)[:, :, 2]
        along = (rel * pad_up).sum(dim=-1)
        return (self._plunger_rest_local_z - along).clamp(0.0, self._plunger_rest_gap)

    @property
    def pad_top_z(self) -> torch.Tensor:
        return self.pad.pose.p[:, 2] + self._collar_height

    def _reset_seal_trackers(self, env_idx: torch.Tensor, b: int) -> None:
        if not hasattr(self, "_seal_sustain_counter"):
            self._seal_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._detent_scale = torch.ones(self.num_envs, device=self.device)
        self._seal_sustain_counter[env_idx] = 0
        self._detent_scale[env_idx] = self._sample_detent_scale(env_idx, b)

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        pad_pos = self.pad.pose.p
        stamp_pos = self.stamp.pose.p

        depth = self._plunger_depth()
        latched = depth >= self._latch_success_depth

        pad_top = self.pad_top_z
        lifted = tcp_pos[:, 2] >= pad_top + self._liftoff_tcp_clearance

        gripped = agent.is_grasping(self.stamp) & (
            torch.linalg.norm(tcp_pos - stamp_pos, dim=-1) < self._stamp_held_distance
        )

        pad_on_belt = (
            (pad_pos[:, 2] > deck_z)
            & (torch.abs(pad_pos[:, 0] - ox) < self._conveyor_half_width + self._pad_half_width)
            & (torch.abs(pad_pos[:, 1] - oy) < self._conveyor_half_length + self._pad_half_width)
        )

        pad_up = quaternion_to_matrix(self.pad.pose.q)[:, :, 2]
        pad_upright = pad_up[:, 2] >= self._pad_upright_cos

        belt_speed = torch.abs(self.belt.velocity)
        along_v = self.pad.linear_velocity[:, 1] * self.belt.direction_sign
        travelling = (along_v >= carried_speed(belt_speed, self._pad_travel_frac)) | (belt_speed <= 1e-6)

        holding = latched & lifted & gripped & pad_on_belt & pad_upright & travelling

        self._seal_sustain_counter = self._advance_sustain("seal_sustain", self._seal_sustain_counter, holding)
        success = self._seal_sustain_counter >= self._seal_sustain_steps

        stamp_axis = quaternion_to_matrix(self.stamp.pose.q)[:, :, 0]
        face_z = stamp_pos[:, 2] + stamp_axis[:, 2] * self._stamp_face_below_tcp
        stamp_face_above_pad = face_z - pad_top

        return {
            "success": success,
            "seal_latched": latched,
            "plunger_depth": depth,
            "stamp_lifted_off": lifted,
            "stamp_gripped": gripped,
            "stamp_face_above_pad": stamp_face_above_pad,
            "seal_sustain": self._seal_sustain_counter,
            "pad_on_belt": pad_on_belt,
            "pad_upright": pad_upright,
            "pad_travelling": travelling,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p

        face_pos = tcp_pos.clone()
        face_pos[:, 2] = face_pos[:, 2] - self._stamp_face_below_tcp
        face_to_bore = torch.linalg.norm(face_pos - self.plunger.pose.p, dim=-1)
        reach_reward = 1 - torch.tanh(5 * face_to_bore)

        press_reward = (info["plunger_depth"] / self._latch_success_depth).clamp(0.0, 1.0)

        lift = (tcp_pos[:, 2] - self.pad_top_z) / self._liftoff_tcp_clearance
        liftoff_reward = lift.clamp(0.0, 1.0) * info["seal_latched"].float()

        return reach_reward + 2.0 * press_reward + liftoff_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0
