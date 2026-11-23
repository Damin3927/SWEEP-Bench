from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_divert_pick.assets import (
    build_divert_block,
    build_divert_plough,
)
from sweep_bench.tasks.conveyor_divert_pick.config import RandomizationConfig
from sweep_bench.tasks.conveyor_divert_pick.scripted import ScriptedDivertPickMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor

_PALM_LINK = {
    "panda": "panda_hand",
    "panda_wristcam": "panda_hand",
}


@bench_task("ConveyorDivertPick-v0")
@register_env("ConveyorDivertPick-v0", max_episode_steps=300)
class ConveyorDivertPickEnv(ScriptedDivertPickMixin, ConveyorWorkspaceEnv):
    """Pick a part while a plough is walking it diagonally across the belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _conveyor_half_width: float = 0.13

    _plough_angle_deg: float = 18.0
    _lane_half: float = 0.07
    _kerb_half_thickness: float = 0.006
    _kerb_half_height: float = 0.0055
    _kerb_ride_gap: float = 0.001
    _kerb_friction: float = 0.10
    _kerb_lead_in: float = 0.10
    _plough_center_dy: float = -0.041

    _rider_half: float = 0.030
    _rider_density: float = 1000.0
    _rider_friction: float = 0.2
    belt_static_friction: float = 1.0
    belt_kinetic_friction: float = 0.8
    _rider_patch_radius: float = 0.03

    _grasp_width_tol: float = 0.008
    _grasp_distance_threshold: float = 0.06
    _carry_speed_match_threshold: float = 0.03
    _lift_clearance: float = 0.14
    _plough_commit_offset: float = 0.05
    _kerb_contact_tol: float = 0.010
    _kerb_contact_force_min: float = 1e-3
    _robot_touch_force_min: float = 1e-3
    _caught_moving_frac: float = 0.5
    _deck_lateral_limit: float = 0.0922
    _pick_sustain_steps: int = 4

    default_prompt: str = "pick up the block as the plough pushes it across the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _plough_angle(self) -> float:
        return math.radians(self._plough_angle_deg)

    @property
    def _plough_functional_length(self) -> float:
        return 2.0 * self._lane_half / math.sin(self._plough_angle)

    @property
    def _plough_half_length(self) -> float:
        return 0.5 * (self._plough_functional_length + self._kerb_lead_in)

    @property
    def _plough_center_dx(self) -> float:
        a = self._plough_angle
        return -(self._kerb_half_thickness + self._rider_half) * math.cos(a) - 0.5 * self._kerb_lead_in * math.sin(a)

    @property
    def _plough_center_z(self) -> float:
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        return deck_z + self._kerb_ride_gap + self._kerb_half_height

    def rail_dir(self, sign: torch.Tensor) -> torch.Tensor:
        a = self._plough_angle
        out = torch.zeros((sign.shape[0], 2), device=self.device)
        out[:, 0] = math.sin(a)
        out[:, 1] = math.cos(a) * sign
        return out

    def rail_normal(self, sign: torch.Tensor) -> torch.Tensor:
        a = self._plough_angle
        out = torch.zeros((sign.shape[0], 2), device=self.device)
        out[:, 0] = math.cos(a)
        out[:, 1] = -math.sin(a) * sign
        return out

    def _yaw(self, sign: torch.Tensor) -> torch.Tensor:
        return -self._plough_angle * sign

    def _yaw_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        q = torch.zeros((yaw.shape[0], 4), device=self.device)
        q[:, 0] = torch.cos(0.5 * yaw)
        q[:, 3] = torch.sin(0.5 * yaw)
        return q

    def _plough_center_xy(self, sign: torch.Tensor) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        out = torch.zeros((sign.shape[0], 2), device=self.device)
        out[:, 0] = ox + self._plough_center_dx
        out[:, 1] = oy + self._plough_center_dy * sign
        return out

    def _plough_pose(self, sign: torch.Tensor) -> Pose:
        xy = self._plough_center_xy(sign)
        pos = torch.zeros((sign.shape[0], 3), device=self.device)
        pos[:, :2] = xy
        pos[:, 2] = self._plough_center_z
        return Pose.create_from_pq(pos, self._yaw_quat(self._yaw(sign)))

    def rail_offset(self, xy: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        d = xy - self._plough_center_xy(sign)
        return (d * self.rail_dir(sign)).sum(dim=-1)

    def kerb_standoff(self, xy: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        d = xy - self._plough_center_xy(sign)
        return (d * self.rail_normal(sign)).sum(dim=-1) - (self._kerb_half_thickness + self._rider_half)

    def lane_diversion(self, xy: torch.Tensor) -> torch.Tensor:
        return (xy[:, 0] - self.workspace_offset[0]) + self._lane_half

    def _kerb_pushes_target(self) -> torch.Tensor:
        force = self.scene.get_pairwise_contact_forces(self.target, self.plough)
        return torch.linalg.norm(force, dim=-1) > self._kerb_contact_force_min

    def _gripper_touches_target(self) -> torch.Tensor:
        agent = cast("Panda", self.agent)
        palm_name = _PALM_LINK.get(self.robot_uids)
        if palm_name is None:
            raise ValueError(
                f"divert-pick has no gripper palm link for robot_uids={self.robot_uids!r}; known: {sorted(_PALM_LINK)}"
            )
        touching = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for link in (agent.finger1_link, agent.finger2_link, agent.robot.links_map[palm_name]):
            force = torch.linalg.norm(self.scene.get_pairwise_contact_forces(link, self.target), dim=-1)
            touching = touching | (force > self._robot_touch_force_min)
        return touching

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        rider_builder = build_divert_block(
            self.scene,
            half_extent=self._rider_half,
            density=self._rider_density,
            friction=self._rider_friction,
        )
        rider_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._rider_half + 0.001])
        self.target = rider_builder.build(name="divert_block")
        self.belt.register_rider(self.target, patch_radius=self._rider_patch_radius)

        plough_builder = build_divert_plough(
            self.scene,
            half_length=self._plough_half_length,
            half_thickness=self._kerb_half_thickness,
            half_height=self._kerb_half_height,
            friction=self._kerb_friction,
        )
        plough_builder.initial_pose = sapien.Pose(
            p=[ox + self._plough_center_dx, oy + self._plough_center_dy, self._plough_center_z]
        )
        self.plough = plough_builder.build_kinematic(name="divert_plough")

        self.lift_height = deck_z + self._lift_clearance

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        sign = self.belt.direction_sign[env_idx]

        self.plough.set_pose(self._plough_pose(sign))

        rider_pos = self._sample_rider_position(env_idx, b)
        self.target.set_pose(Pose.create_from_pq(rider_pos, self._yaw_quat(self._yaw(sign))))
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        self.target.set_linear_velocity(vel)
        self.target.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_pick_trackers(env_idx)

    def _sample_rider_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        device = self.device
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        sign = self.belt.direction_sign[env_idx]
        margin = self._rider_half * (math.cos(self._plough_angle) + math.sin(self._plough_angle)) + 0.02
        start_y = self._conveyor_half_length - margin
        if self.randomization.lane_phase:
            lo, hi = self.randomization.lane_phase_range
            phase = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
            start_y = start_y + phase
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox - self._lane_half
        pos[:, 1] = oy - start_y * sign
        pos[:, 2] = deck_z + self._rider_half + 0.001
        return pos

    def _reset_pick_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_pick_sustain"):
            zeros = torch.zeros(self.num_envs, device=self.device)
            self._pick_sustain = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._commit_diversion = zeros.clone()
            self._commit_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._commit_on_kerb = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._plough_diversion = zeros.clone()
            self._prev_diversion = zeros.clone()
            self._was_grasping = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._pre_grasp_carried = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._caught_moving = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._left_deck = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._carry_speed = zeros.clone()
            self._prev_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._prev_tcp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._pick_sustain[env_idx] = 0
        self._commit_diversion[env_idx] = 0.0
        self._commit_latched[env_idx] = False
        self._commit_on_kerb[env_idx] = False
        self._plough_diversion[env_idx] = 0.0
        self._prev_diversion[env_idx] = self.lane_diversion(self.target.pose.p[:, :2])[env_idx]
        self._was_grasping[env_idx] = False
        self._pre_grasp_carried[env_idx] = True
        self._caught_moving[env_idx] = True
        self._left_deck[env_idx] = False
        self._carry_speed[env_idx] = math.inf
        self._prev_target_pos[env_idx] = self.target.pose.p[env_idx]
        self._prev_tcp_pos[env_idx] = cast("Panda", self.agent).tcp.pose.p[env_idx]

    @property
    def _belt_push_watch(self) -> Actor:
        return self.target

    def evaluate(self):
        ox, _oy, _oz = self.workspace_offset
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        target_pos = self.target.pose.p
        sign = self.belt.direction_sign
        advanced = self._eval_advanced("divert_pick_step")

        finger_width = self.gripper_aperture()
        width_matches = torch.abs(finger_width - self._rider_grasp_width) <= self._grasp_width_tol
        rel = (target_pos - tcp_pos) - (self._prev_target_pos - self._prev_tcp_pos)
        step_speed = torch.linalg.norm(rel, dim=-1) * self.control_freq
        self._carry_speed = torch.where(advanced, step_speed, self._carry_speed)
        self._prev_target_pos = torch.where(advanced.unsqueeze(-1), target_pos, self._prev_target_pos)
        self._prev_tcp_pos = torch.where(advanced.unsqueeze(-1), tcp_pos, self._prev_tcp_pos)
        carried = self._carry_speed < self._carry_speed_match_threshold
        co_located = torch.linalg.norm(tcp_pos - target_pos, dim=-1) < self._grasp_distance_threshold
        grasping_now = width_matches & agent.is_grasping(self.target)
        is_held = grasping_now & carried & co_located

        is_lifted = target_pos[:, 2] > self.lift_height

        diversion = self.lane_diversion(target_pos[:, :2])
        riding_kerb = self._kerb_pushes_target()
        robot_touching = self._gripper_touches_target()
        gained = (diversion - self._prev_diversion).clamp(min=0.0)
        banked = self._plough_diversion + torch.where(
            riding_kerb & ~is_held & ~robot_touching & advanced, gained, torch.zeros_like(gained)
        )
        self._plough_diversion = torch.where(advanced, torch.minimum(banked, diversion.clamp(min=0.0)), banked)
        self._prev_diversion = torch.where(advanced, diversion, self._prev_diversion)

        carried_by_belt = (
            self.belt_carried(self.target, self._caught_moving_frac)
            & self.riding_on_deck(self.target, self._rider_half)
            & ~self.gripper_drove_along_belt()
        )
        self._pre_grasp_carried = torch.where(advanced & ~grasping_now, carried_by_belt, self._pre_grasp_carried)
        first_grasp = grasping_now & ~self._was_grasping
        self._caught_moving = torch.where(first_grasp, self._pre_grasp_carried, self._caught_moving)
        rail_s = self.rail_offset(target_pos[:, :2], sign)
        against_kerb = (torch.abs(rail_s) <= self._plough_half_length) & (
            torch.abs(self.kerb_standoff(target_pos[:, :2], sign)) <= self._kerb_contact_tol
        )
        self._commit_diversion = torch.where(first_grasp, diversion, self._commit_diversion)
        self._commit_on_kerb = torch.where(first_grasp, against_kerb, self._commit_on_kerb)
        self._commit_latched = self._commit_latched | grasping_now
        self._was_grasping = torch.where(advanced, grasping_now, self._was_grasping)
        diverted = (
            self._commit_latched
            & (self._plough_diversion >= self._plough_commit_offset)
            & (self._commit_diversion >= self._plough_commit_offset)
            & self._commit_on_kerb
        )

        off_deck_now = (torch.abs(target_pos[:, 0] - ox) > self._deck_lateral_limit) & (
            target_pos[:, 2] <= self.lift_height
        )
        self._left_deck = self._left_deck | (off_deck_now & advanced)
        on_deck = ~self._left_deck

        v_xy = self.target.linear_velocity[:, :2]
        rider_speed = torch.linalg.norm(self.target.linear_velocity, dim=-1)
        on_rail = torch.abs(self.rail_offset(target_pos[:, :2], sign)) <= self._plough_half_length
        along = (v_xy[:, 1] * sign).clamp(min=0.0)
        divert_angle = torch.rad2deg(torch.atan2(torch.abs(v_xy[:, 0]), along.clamp(min=1e-6)))
        divert_angle = torch.where(
            on_rail & (torch.linalg.norm(v_xy, dim=-1) > 0.02) & ~is_held,
            divert_angle,
            torch.zeros_like(divert_angle),
        )

        picked = is_held & is_lifted & diverted & on_deck & self._caught_moving
        self._pick_sustain = self._advance_sustain("divert_pick", self._pick_sustain, picked)
        success = self._pick_sustain >= self._pick_sustain_steps

        return {
            "success": success,
            "is_held": is_held,
            "caught_moving": self._caught_moving,
            "is_lifted": is_lifted,
            "diverted": diverted,
            "on_deck": on_deck,
            "rider_speed": rider_speed,
            "commit_diversion": self._commit_diversion,
            "commit_on_kerb": self._commit_on_kerb,
            "plough_diversion": self._plough_diversion,
            "lane_diversion": diversion,
            "divert_angle_deg": divert_angle,
            "rail_offset": self.rail_offset(target_pos[:, :2], sign),
            "pick_sustain": self._pick_sustain,
        }

    @property
    def _rider_grasp_width(self) -> float:
        return 2.0 * self._rider_half

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        target_pos = self.target.pose.p

        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - target_pos, dim=-1))
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        lift_reward = torch.clamp((target_pos[:, 2] - deck_z) / self._lift_clearance, 0, 1)
        credit = (
            info["is_held"].float() * info["diverted"].float() * info["on_deck"].float() * info["caught_moving"].float()
        )
        return reach_reward + 2.0 * lift_reward * credit

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
