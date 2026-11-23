from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_multiply,
    quaternion_to_matrix,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_swing_load.assets import build_carry_frame, build_pendant
from sweep_bench.tasks.conveyor_swing_load.config import RandomizationConfig
from sweep_bench.tasks.conveyor_swing_load.scripted import ScriptedSwingLoadMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@bench_task("ConveyorSwingLoad-v0")
@register_env("ConveyorSwingLoad-v0", max_episode_steps=300)
class ConveyorSwingLoadEnv(ScriptedSwingLoadMixin, ConveyorWorkspaceEnv):
    """Lift a carry frame off the belt and bring its hanging pendant to rest."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _foot_half_x: float = 0.050
    _foot_half_y: float = 0.050
    _foot_half_thickness: float = 0.005
    _post_inner_x: float = 0.034
    _post_outer_x: float = 0.050
    _post_inner_y: float = 0.038
    _post_outer_y: float = 0.050
    _rail_half_x: float = 0.016
    _rail_half_y: float = 0.050
    _rail_half_thickness: float = 0.006
    _rail_center_z: float = 0.164
    _pin_z: float = 0.140
    _pin_radius: float = 0.002
    _pin_clearance: float = 0.0003
    _journal_inner_x: float = 0.004
    _journal_outer_x: float = 0.013
    _wall_half_thickness: float = 0.0015
    _frame_density: float = 500.0
    _bearing_friction: float = 0.01

    _pin_half_length: float = 0.0135
    _collar_radius: float = 0.005
    _collar_half_length: float = 0.0015
    _collar_x: float = 0.0155
    _stem_half_x: float = 0.003
    _stem_half_y: float = 0.005
    _stem_half_length: float = 0.050
    _bob_half: tuple[float, float, float] = (0.020, 0.010, 0.010)
    _bob_center_z: float = -0.110
    _stem_density: float = 1000.0
    _bob_density: float = 3000.0
    _pendant_angular_damping: float = 0.0

    _frame_patch_radius: float = 0.040

    _spawn_end_margin: float = 0.04

    _lift_clearance: float = 0.06
    _grip_gap_squeeze_max: float = 0.014
    _grip_gap_slack_max: float = 0.014
    _pendant_rest_angle: float = math.radians(4.0)
    _pendant_period: float = 0.667
    _pendant_rate_window: int = 1
    _frame_carry_speed_match: float = 0.04
    _frame_arrest_speed: float = 0.05
    _caught_moving_frac: float = 0.5
    _pin_capture: float = 0.012
    _sustain_steps: int = 8

    default_prompt: str = "lift the carry frame off the conveyor belt and steady its hanging weight"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def grasp_width(self) -> float:
        return 2.0 * self._rail_half_x

    @property
    def _frame_top_z(self) -> float:
        return self._rail_center_z + self._rail_half_thickness

    def frame_lowest_z(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.frame.pose.q)
        drop = (
            rot[:, 2, 0].abs() * self._foot_half_x
            + rot[:, 2, 1].abs() * self._foot_half_y
            - self._frame_top_z * rot[:, 2, 2].clamp(max=0.0)
        )
        return self.frame.pose.p[:, 2] - drop

    @property
    def _journal_cap_top_z(self) -> float:
        return self._pin_z + self._pin_radius + self._pin_clearance + 2.0 * self._wall_half_thickness

    @property
    def _pendant_bottom_z(self) -> float:
        return self._bob_center_z - self._bob_half[2]

    @property
    def _pendant_deck_clearance(self) -> float:
        return self._pin_z + self._pendant_bottom_z

    @property
    def pendant_rest_rate(self) -> float:
        return 2.0 * math.pi / self._pendant_period * self._pendant_rest_angle

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        frame_builder = build_carry_frame(
            self.scene,
            foot_half_x=self._foot_half_x,
            foot_half_y=self._foot_half_y,
            foot_half_thickness=self._foot_half_thickness,
            post_inner_x=self._post_inner_x,
            post_outer_x=self._post_outer_x,
            post_inner_y=self._post_inner_y,
            post_outer_y=self._post_outer_y,
            rail_half_x=self._rail_half_x,
            rail_half_y=self._rail_half_y,
            rail_half_thickness=self._rail_half_thickness,
            rail_center_z=self._rail_center_z,
            pin_z=self._pin_z,
            pin_radius=self._pin_radius,
            pin_clearance=self._pin_clearance,
            journal_inner_x=self._journal_inner_x,
            journal_outer_x=self._journal_outer_x,
            wall_half_thickness=self._wall_half_thickness,
            density=self._frame_density,
            bearing_friction=self._bearing_friction,
        )
        frame_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z])
        self.frame = frame_builder.build(name="carry_frame")
        self.belt.register_rider(self.frame, patch_radius=self._frame_patch_radius)

        pendant_builder = build_pendant(
            self.scene,
            pin_radius=self._pin_radius,
            pin_half_length=self._pin_half_length,
            collar_radius=self._collar_radius,
            collar_half_length=self._collar_half_length,
            collar_x=self._collar_x,
            stem_half_x=self._stem_half_x,
            stem_half_y=self._stem_half_y,
            stem_half_length=self._stem_half_length,
            bob_half=self._bob_half,
            bob_center_z=self._bob_center_z,
            stem_density=self._stem_density,
            bob_density=self._bob_density,
            bearing_friction=self._bearing_friction,
        )
        pendant_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._pin_z])
        self.pendant = pendant_builder.build(name="pendant")
        self.pendant.angular_damping = self._pendant_angular_damping

    @property
    def _primary_rider(self) -> Actor:
        return self.pendant

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        frame_pos = self._sample_frame_position(env_idx, b)
        upright = torch.zeros((b, 4), device=device)
        upright[:, 0] = 1.0
        self.frame.set_pose(Pose.create_from_pq(frame_pos, upright))

        pendant_pos = frame_pos.clone()
        pendant_pos[:, 2] = frame_pos[:, 2] + self._pin_z
        self.pendant.set_pose(Pose.create_from_pq(pendant_pos, self._sample_pendant_orientation(upright, b)))

        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        zero = torch.zeros((b, 3), device=device)
        self.frame.set_linear_velocity(vel)
        self.frame.set_angular_velocity(zero)
        self.pendant.set_linear_velocity(vel.clone())
        self.pendant.set_angular_velocity(zero.clone())

        self._reset_swing_trackers(env_idx)

    def _sample_pendant_orientation(self, frame_q: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        swing = torch.zeros(b, device=device)
        if self.randomization.pendant_start_swing:
            lo, hi = self.randomization.swing_range
            swing_np = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            swing = torch.as_tensor(swing_np, device=device)
        hinge_q = torch.zeros((b, 4), device=device)
        hinge_q[:, 0] = torch.cos(swing * 0.5)
        hinge_q[:, 1] = torch.sin(swing * 0.5)
        return quaternion_multiply(frame_q, hinge_q)

    def _sample_frame_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._foot_half_y - self._spawn_end_margin
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z
        return pos

    def pendant_in_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        rot_frame = quaternion_to_matrix(self.frame.pose.q)
        rot_pendant = quaternion_to_matrix(self.pendant.pose.q)
        rel = torch.bmm(rot_frame.transpose(1, 2), rot_pendant)
        angle = torch.atan2(-rel[:, 1, 2], rel[:, 2, 2])

        anchor = torch.tensor([0.0, 0.0, self._pin_z], device=self.device)
        pin_in_frame = torch.bmm(
            rot_frame.transpose(1, 2), (self.pendant.pose.p - self.frame.pose.p).unsqueeze(-1)
        ).squeeze(-1)
        return angle, torch.linalg.norm(pin_in_frame - anchor, dim=-1)

    def pendant_swing_rate(self) -> torch.Tensor:
        hist = self._pendant_angle_hist
        return (hist[:, -1] - hist[:, 0]) * self.control_freq / self._pendant_rate_window

    def _reset_swing_trackers(self, env_idx: torch.Tensor) -> None:
        window = self._pendant_rate_window
        if not hasattr(self, "_swing_sustain_counter"):
            self._swing_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._pin_ever_lost = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._pendant_angle_hist = torch.zeros((self.num_envs, window + 1), device=self.device)
            self._prev_frame_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._prev_tcp_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._was_holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._pre_grasp_carried = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self._caught_moving = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._swing_sustain_counter[env_idx] = 0
        self._pin_ever_lost[env_idx] = False
        self._was_holding[env_idx] = False
        self._pre_grasp_carried[env_idx] = True
        self._caught_moving[env_idx] = True
        angle, _offset = self.pendant_in_frame()
        self._pendant_angle_hist[env_idx] = angle[env_idx].unsqueeze(-1)
        self._prev_frame_pos[env_idx] = self.frame.pose.p[env_idx]
        self._prev_tcp_pos[env_idx] = cast("Panda", self.agent).tcp.pose.p[env_idx]

    @property
    def _belt_push_watch(self) -> Actor:
        return self.frame

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        frame_pos = self.frame.pose.p
        advanced = self._eval_advanced("swing_load_step")

        angle, pin_offset = self.pendant_in_frame()
        rolled = torch.cat([self._pendant_angle_hist[:, 1:], angle.unsqueeze(-1)], dim=1)
        self._pendant_angle_hist = torch.where(advanced.unsqueeze(-1), rolled, self._pendant_angle_hist)
        swing_rate = self.pendant_swing_rate()

        frame_rel_tcp = ((frame_pos - tcp_pos) - (self._prev_frame_pos - self._prev_tcp_pos)) * self.control_freq
        frame_world = (frame_pos - self._prev_frame_pos) * self.control_freq
        self._prev_frame_pos = torch.where(advanced.unsqueeze(-1), frame_pos, self._prev_frame_pos)
        self._prev_tcp_pos = torch.where(advanced.unsqueeze(-1), tcp_pos, self._prev_tcp_pos)

        finger_gap = self.gripper_aperture()
        width = self.grasp_width
        width_matches = (finger_gap > width - self._grip_gap_squeeze_max) & (
            finger_gap < width + self._grip_gap_slack_max
        )
        frame_held = width_matches & agent.is_grasping(self.frame)

        carried_by_belt = (
            self.belt_carried(self.frame, self._caught_moving_frac)
            & self.riding_on_deck(self.frame, 0.0)
            & ~self.gripper_drove_along_belt()
        )
        self._pre_grasp_carried = torch.where(advanced & ~frame_held, carried_by_belt, self._pre_grasp_carried)
        first_hold = frame_held & ~self._was_holding
        self._caught_moving = torch.where(first_hold, self._pre_grasp_carried, self._caught_moving)
        self._was_holding = torch.where(advanced, frame_held, self._was_holding)

        is_lifted = self.frame_lowest_z() > deck_z + self._lift_clearance

        angle_ok = torch.abs(angle) <= self._pendant_rest_angle
        rate_ok = torch.abs(swing_rate) <= self.pendant_rest_rate
        pendant_at_rest = angle_ok & rate_ok

        is_carried = torch.linalg.norm(frame_rel_tcp, dim=-1) < self._frame_carry_speed_match

        is_arrested = torch.linalg.norm(frame_world, dim=-1) < self._frame_arrest_speed

        over_belt = (torch.abs(frame_pos[:, 0] - ox) <= self._conveyor_half_width) & (
            torch.abs(frame_pos[:, 1] - oy) <= self._conveyor_half_length
        )

        self._pin_ever_lost = self._pin_ever_lost | ((pin_offset >= self._pin_capture) & advanced)
        pin_captured = ~self._pin_ever_lost

        holding = (
            frame_held
            & is_lifted
            & pendant_at_rest
            & is_carried
            & is_arrested
            & over_belt
            & pin_captured
            & self._caught_moving
        )
        self._swing_sustain_counter = self._advance_sustain("swing_sustain", self._swing_sustain_counter, holding)

        return {
            "success": self._swing_sustain_counter >= self._sustain_steps,
            "frame_held": frame_held,
            "is_lifted": is_lifted,
            "pendant_angle": angle,
            "pendant_rate": swing_rate,
            "angle_ok": angle_ok,
            "rate_ok": rate_ok,
            "pendant_at_rest": pendant_at_rest,
            "is_carried": is_carried,
            "caught_moving": self._caught_moving,
            "is_arrested": is_arrested,
            "over_belt": over_belt,
            "pin_offset": pin_offset,
            "pin_captured": pin_captured,
            "finger_gap": finger_gap,
            "swing_sustain": self._swing_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        rail_world_z = self.frame.pose.p[:, 2] + self._rail_center_z
        rail_pos = torch.cat([self.frame.pose.p[:, :2], rail_world_z.unsqueeze(-1)], dim=-1)
        reaching_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - rail_pos, dim=-1))

        lift_reward = torch.clamp((self.frame.pose.p[:, 2] - deck_z) / self._lift_clearance, 0, 1)
        settle = (1 - torch.tanh(torch.abs(info["pendant_angle"]) / self._pendant_rest_angle)) * (
            1 - torch.tanh(torch.abs(info["pendant_rate"]) / self.pendant_rest_rate)
        )
        credit = (
            info["frame_held"].float()
            * info["is_lifted"].float()
            * info["pin_captured"].float()
            * info["over_belt"].float()
            * info["is_arrested"].float()
            * info["is_carried"].float()
            * info["caught_moving"].float()
        )
        return reaching_reward + 2.0 * lift_reward * credit + 2.0 * settle * credit

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
