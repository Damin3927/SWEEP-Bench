from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_match_angle.assets import (
    build_angle_display,
    build_arrow_puck,
    build_round_cradle,
)
from sweep_bench.tasks.conveyor_match_angle.config import RandomizationConfig
from sweep_bench.tasks.conveyor_match_angle.scripted import ScriptedMatchAngleMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor

_PUCK_COLOR = (0.88, 0.86, 0.80, 1.0)
_MARK_COLOR = (0.95, 0.38, 0.08, 1.0)
_CARRIER_COLOR = (0.40, 0.43, 0.47, 1.0)
_CRADLE_COLOR = (0.24, 0.26, 0.30, 1.0)
_DISPLAY_POST_COLOR = (0.30, 0.31, 0.33, 1.0)
_DISPLAY_VANE_COLOR = (0.90, 0.90, 0.88, 1.0)


def _wrap_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _body_y_xy(quat: torch.Tensor) -> torch.Tensor:
    from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

    return quaternion_to_matrix(quat)[:, :2, 1]


def _yaw_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(dim=-1))


@bench_task("ConveyorMatchAngle-v0")
@register_env("ConveyorMatchAngle-v0", max_episode_steps=300)
class ConveyorMatchAngleEnv(ScriptedMatchAngleMixin, ConveyorWorkspaceEnv):
    """Set a marked puck into a ROUND cradle on the belt, aimed at a display arrow."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _puck_radius: float = 0.022
    _puck_half_height: float = 0.015
    _puck_grip_offset: float = 0.014

    _carrier_half_x: float = 0.052
    _carrier_half_y: float = 0.080
    _carrier_half_z: float = 0.004
    _cradle_inner_radius: float = 0.026
    _cradle_wall_half_thickness: float = 0.003
    _cradle_wall_half_height: float = 0.009
    _cradle_wall_tilt_deg: float = 42.0
    _cradle_segments: int = 12
    _carrier_patch_radius: float = 0.050

    _display_dx: float = 0.24
    _display_dy: float = 0.18
    _display_post_half_width: float = 0.018
    _display_post_half_height: float = 0.135
    _display_vane_radius: float = 0.055
    _display_vane_half_thickness: float = 0.004
    _display_x_range: tuple[float, float] = (0.21, 0.27)
    _display_y_range: tuple[float, float] = (-0.22, 0.22)

    _mark_offset_range: tuple[float, float] = (-math.pi * 25.0 / 180.0, math.pi * 25.0 / 180.0)

    _angle_tol: float = math.radians(12.0)
    _seat_z_carrier: float = _carrier_half_z + _puck_half_height
    _seat_depth_tol: float = 0.004
    _seat_lateral_tol: float = 0.006
    _released_finger_width: float = 2.0 * _puck_radius + 0.010
    _released_distance_threshold: float = 0.08
    _comove_tol: float = 0.05
    _carrier_x_tol: float = 0.04
    _carrier_comove_tol: float = 0.08
    _carrier_tilt_cos: float = math.cos(math.radians(10.0))
    _carrier_y_limit: float = 0.85
    _match_sustain_steps: int = 8

    default_prompt: str = "put the puck in the round cradle with its arrow pointing the same way as the display arrow"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _cradle_rim_local_z(self) -> float:
        return self._carrier_half_z + 2.0 * self._cradle_wall_half_height * math.cos(
            math.radians(self._cradle_wall_tilt_deg)
        )

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        carrier_builder = build_round_cradle(
            self.scene,
            plate_half_x=self._carrier_half_x,
            plate_half_y=self._carrier_half_y,
            plate_half_z=self._carrier_half_z,
            inner_radius=self._cradle_inner_radius,
            wall_half_thickness=self._cradle_wall_half_thickness,
            wall_half_height=self._cradle_wall_half_height,
            wall_tilt_deg=self._cradle_wall_tilt_deg,
            n_segments=self._cradle_segments,
            plate_color=_CARRIER_COLOR,
            wall_color=_CRADLE_COLOR,
        )
        carrier_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._carrier_half_z])
        self.carrier = carrier_builder.build(name="cradle_carrier")
        self.belt.register_rider(self.carrier, patch_radius=self._carrier_patch_radius)

        puck_builder = build_arrow_puck(
            self.scene,
            radius=self._puck_radius,
            half_height=self._puck_half_height,
            body_color=_PUCK_COLOR,
            mark_color=_MARK_COLOR,
        )
        puck_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.puck = puck_builder.build(name="arrow_puck")

        display_builder = build_angle_display(
            self.scene,
            post_half_width=self._display_post_half_width,
            post_half_height=self._display_post_half_height,
            vane_radius=self._display_vane_radius,
            vane_half_thickness=self._display_vane_half_thickness,
            post_color=_DISPLAY_POST_COLOR,
            vane_color=_DISPLAY_VANE_COLOR,
            mark_color=_MARK_COLOR,
        )
        display_builder.initial_pose = sapien.Pose(
            p=[ox + self._display_dx, oy + self._display_dy, oz + 2.0 * self._display_post_half_height]
        )
        self.display = display_builder.build_kinematic(name="angle_display")

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._puck_radius - 0.001, dtype=np.float32)

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.display]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        self._reset_match_trackers(env_idx)

        display_yaw = self._sample_display_yaw(b)
        display_pos = self._sample_display_position(b)
        self.display.set_pose(Pose.create_from_pq(display_pos, _yaw_quat(display_yaw)))

        carrier_pos = self._sample_carrier_position(env_idx, b)
        carrier_q = _yaw_quat(self._sample_carrier_yaw(b))
        self.carrier.set_pose(Pose.create_from_pq(carrier_pos, carrier_q))
        carrier_vel = torch.zeros((b, 3), device=device)
        carrier_vel[:, 1] = self.belt.velocity[env_idx]
        self.carrier.set_linear_velocity(carrier_vel)
        self.carrier.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pose = cast("Panda", self.agent).tcp.pose
        tcp_b = Pose.create_from_pq(tcp_pose.p[env_idx], tcp_pose.q[env_idx])

        mark_offset = self._sample_mark_offset(b)
        base_q = self._gripper_down_base_q(env_idx)
        yaw_cmd = _yaw_between(_body_y_xy(base_q), _body_y_xy(tcp_b.q)) - mark_offset
        mount = self._mount_in_gripper(base_q, display_yaw - yaw_cmd)
        local = torch.zeros((b, 3), device=device)
        local[:, 2] = self._puck_grip_offset
        puck_pose = tcp_b * Pose.create_from_pq(local, mount)
        self.puck.set_pose(puck_pose)
        self.puck.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.puck.set_angular_velocity(torch.zeros((b, 3), device=device))
        self._mark_offset[env_idx] = mark_offset
        self._scripted_yaw_cmd[env_idx] = yaw_cmd

    def _mount_in_gripper(self, base_q: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import (
            quaternion_invert,
            quaternion_multiply,
        )

        return quaternion_multiply(quaternion_invert(base_q), _yaw_quat(heading))

    def _gripper_down_base_q(self, env_idx: torch.Tensor) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion

        b = len(env_idx)
        sign = self.belt.direction_sign[env_idx]
        y_body = torch.zeros((b, 3), device=self.device)
        y_body[:, 0] = sign
        z_body = torch.zeros((b, 3), device=self.device)
        z_body[:, 2] = -1.0
        x_body = torch.cross(y_body, z_body, dim=-1)
        return matrix_to_quaternion(torch.stack([x_body, y_body, z_body], dim=-1))

    def _sample_display_yaw(self, b: int) -> torch.Tensor:
        if not self.randomization.reference_yaw:
            return torch.zeros(b, device=self.device, dtype=torch.float32)
        lo, hi = self.randomization.display_yaw_range
        return torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)

    def _sample_mark_offset(self, b: int) -> torch.Tensor:
        lo, hi = self._mark_offset_range
        return torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)

    def _sample_carrier_yaw(self, b: int) -> torch.Tensor:
        if not self.randomization.carrier_yaw:
            return torch.zeros(b, device=self.device, dtype=torch.float32)
        lo, hi = self.randomization.carrier_yaw_range
        return torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)

    def _sample_display_position(self, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        pos = torch.zeros((b, 3), device=self.device)
        pos[:, 0] = ox + self._display_dx
        pos[:, 1] = oy + self._display_dy
        pos[:, 2] = oz + 2.0 * self._display_post_half_height
        if self.randomization.display_position:
            xs = self._episode_rng.uniform(*self._display_x_range, size=b).astype(np.float32)
            ys = self._episode_rng.uniform(*self._display_y_range, size=b).astype(np.float32)
            pos[:, 0] = ox + torch.as_tensor(xs, device=self.device)
            pos[:, 1] = oy + torch.as_tensor(ys, device=self.device)
        return pos

    def _sample_carrier_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        start_y = self._conveyor_half_length - self._carrier_half_y - 0.04
        pos = torch.zeros((b, 3), device=self.device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._carrier_half_z
        return pos

    def _reset_match_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_match_sustain_counter"):
            self._match_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._mark_offset = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self._scripted_yaw_cmd = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._match_sustain_counter[env_idx] = 0

    def _heading(self, quat: torch.Tensor) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

        axis = quaternion_to_matrix(quat)[:, :2, 0]
        return torch.atan2(axis[:, 1], axis[:, 0])

    def _angle_error(self) -> torch.Tensor:
        return torch.abs(_wrap_pi(self._heading(self.puck.pose.q) - self._heading(self.display.pose.q)))

    def evaluate(self):
        agent = cast("Panda", self.agent)
        carrier_pose = self.carrier.pose
        carrier_pos = carrier_pose.p
        puck_pos = self.puck.pose.p
        belt_v = self.belt.velocity

        angle_err = self._angle_error()
        angle_ok = angle_err < self._angle_tol

        local = (carrier_pose.inv() * self.puck.pose).p
        seat_lateral = torch.linalg.norm(local[:, :2], dim=-1)
        in_pocket = seat_lateral < self._seat_lateral_tol
        at_depth = torch.abs(local[:, 2] - self._seat_z_carrier) < self._seat_depth_tol

        finger_width = self.gripper_aperture()
        released = (finger_width >= self._released_finger_width) & (
            torch.linalg.norm(agent.tcp.pose.p - puck_pos, dim=-1) > self._released_distance_threshold
        )

        comoving = (
            torch.linalg.norm(self.puck.linear_velocity - self.carrier.linear_velocity, dim=-1) < self._comove_tol
        )

        ox, oy, _oz = self.workspace_offset
        up = torch.zeros((self.num_envs, 3), device=self.device)
        up[:, 2] = 1.0
        tilt_cos = ((carrier_pose * Pose.create_from_pq(up)).p - carrier_pos)[:, 2]
        carrier_on_belt = (
            (torch.abs(carrier_pos[:, 0] - ox) <= self._carrier_x_tol)
            & (torch.abs(self.carrier.linear_velocity[:, 1] - belt_v) <= self._carrier_comove_tol)
            & (tilt_cos >= self._carrier_tilt_cos)
            & (torch.abs(carrier_pos[:, 1] - oy) <= self._carrier_y_limit)
        )

        matched = angle_ok & in_pocket & at_depth & released & comoving & carrier_on_belt

        self._match_sustain_counter = self._advance_sustain("match_sustain", self._match_sustain_counter, matched)
        success = self._match_sustain_counter >= self._match_sustain_steps

        return {
            "success": success,
            "angle_ok": angle_ok,
            "angle_error": angle_err,
            "in_pocket": in_pocket,
            "at_depth": at_depth,
            "released": released,
            "comoving": comoving,
            "carrier_on_belt": carrier_on_belt,
            "match_sustain": self._match_sustain_counter,
            "seat_lateral": seat_lateral,
            "mark_offset": self._mark_offset,
        }

    def compute_dense_reward(self, obs, action, info):
        pocket_xy = self.carrier.pose.p[:, :2]
        xy_dist = torch.linalg.norm(self.puck.pose.p[:, :2] - pocket_xy, dim=-1)
        carry_reward = 1 - torch.tanh(8 * xy_dist)
        aim_reward = (1.0 - info["angle_error"] / (math.pi / 2.0)).clamp(0.0, 1.0)
        seated = (info["in_pocket"] & info["at_depth"] & info["angle_ok"]).float()
        return carry_reward + aim_reward + 2.0 * seated

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    q = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=torch.float32)
    q[:, 0] = torch.cos(0.5 * yaw)
    q[:, 3] = torch.sin(0.5 * yaw)
    return q
