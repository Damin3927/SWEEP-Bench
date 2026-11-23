from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import tolerance_speed
from sweep_bench.tasks.conveyor_hook_rail.assets import build_hook_rack, build_rail_saddle
from sweep_bench.tasks.conveyor_hook_rail.config import RandomizationConfig
from sweep_bench.tasks.conveyor_hook_rail.scripted import ScriptedHookRailMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_GRIP_OFFSET_Q = (0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0)

_FINGER_FACE_OFFSET = 0.0032


@bench_task("ConveyorHookRail-v0")
@register_env("ConveyorHookRail-v0", max_episode_steps=300)
class ConveyorHookRailEnv(ScriptedHookRailMixin, ConveyorWorkspaceEnv):
    """Hang a held open-throat saddle on the cross-bar of a rack riding the belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _base_half_x: float = 0.050
    _base_half_y: float = 0.088
    _base_half_height: float = 0.006
    _upright_half_x: float = 0.010
    _upright_half_y: float = 0.010
    _upright_half_height: float = 0.045
    _upright_span_half: float = 0.075
    _bar_radius: float = 0.005
    _bar_half_length: float = 0.075
    _bar_land_half_x: float = 0.004
    _rack_patch_radius: float = 0.035

    _seat_local_z: float = 0.016
    _throat_half_width: float = 0.010
    _throat_depth: float = 0.020
    _wall_thickness: float = 0.010
    _saddle_half_y: float = 0.011
    _lip_flare: float = 0.004
    _lip_height: float = 0.008
    _roof_half_height: float = 0.0015

    _along_bar_tol: float = 0.020
    _hang_below_bar_min: float = 0.005
    _released_finger_width: float = 0.035
    _tcp_clearance_above_bar: float = 0.10
    _settled_vz_tol: float = 0.05
    _comove_rel_tol: float = 0.3
    _rack_x_tol: float = 0.030
    _rack_comove_tol_frac: float = 0.25
    _rack_comove_tol_floor: float = 0.020
    _rack_comove_tol_ceiling: float = 0.050
    _rack_tilt_cos: float = math.cos(math.radians(10.0))
    _finger_face_offset: float = _FINGER_FACE_OFFSET
    _hang_sustain_steps: int = 10

    default_prompt: str = "hang the saddle on the rack's rail on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _bar_above_rack_origin(self) -> float:
        return self._base_half_height + 2.0 * self._upright_half_height

    @property
    def _cradle_rise(self) -> float:
        return self._bar_radius + self._bar_land_half_x

    @property
    def _saddle_grasp_half(self) -> float:
        return self._throat_half_width + self._wall_thickness

    @property
    def _saddle_bottom_local_z(self) -> float:
        return self._seat_local_z - self._throat_depth - self._lip_height

    @property
    def _throat_clear(self) -> float:
        return self._throat_half_width - self._bar_radius

    @property
    def _throat_top_local(self) -> float:
        return self._seat_local_z + self._cradle_rise

    @property
    def _throat_bottom_local(self) -> float:
        return self._seat_local_z - self._throat_depth

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        rack_builder = build_hook_rack(
            self.scene,
            base_half_x=self._base_half_x,
            base_half_y=self._base_half_y,
            base_half_height=self._base_half_height,
            upright_half_x=self._upright_half_x,
            upright_half_y=self._upright_half_y,
            upright_half_height=self._upright_half_height,
            upright_span_half=self._upright_span_half,
            bar_radius=self._bar_radius,
            bar_half_length=self._bar_half_length,
            land_half_x=self._bar_land_half_x,
        )
        rack_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._base_half_height + 0.001])
        self.rack = rack_builder.build(name="hook_rack")
        self.belt.register_rider(self.rack, patch_radius=self._rack_patch_radius)

        saddle_builder = build_rail_saddle(
            self.scene,
            seat_local_z=self._seat_local_z,
            throat_half_width=self._throat_half_width,
            throat_depth=self._throat_depth,
            wall_thickness=self._wall_thickness,
            half_y=self._saddle_half_y,
            cradle_rise=self._cradle_rise,
            lip_flare=self._lip_flare,
            lip_height=self._lip_height,
            roof_half_height=self._roof_half_height,
        )
        saddle_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.saddle = saddle_builder.build(name="rail_saddle")

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._saddle_grasp_half - _FINGER_FACE_OFFSET - 0.001, dtype=np.float32)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        rack_pos = self._sample_rack_position(env_idx, b)
        rack_q = self._sample_rack_orientation(b)
        self.rack.set_pose(Pose.create_from_pq(rack_pos, rack_q))
        rack_vel = torch.zeros((b, 3), device=device)
        rack_vel[:, 1] = self.belt.velocity[env_idx]
        self.rack.set_linear_velocity(rack_vel)
        self.rack.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pose = cast("Panda", self.agent).tcp.pose
        offset_q = torch.tensor(_GRIP_OFFSET_Q, dtype=torch.float32, device=device).expand(b, 4)
        saddle_q = quaternion_multiply(tcp_pose.q[env_idx], offset_q)
        self.saddle.set_pose(Pose.create_from_pq(tcp_pose.p[env_idx], saddle_q))
        self.saddle.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.saddle.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_hang_trackers(env_idx)

    def _sample_rack_orientation(self, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.rack_yaw:
            lo, hi = self.randomization.yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
            q[:, 0] = torch.cos(0.5 * yaw)
            q[:, 3] = torch.sin(0.5 * yaw)
        return q

    def _sample_rack_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._base_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.rack_x_jitter:
            lo, hi = self.randomization.x_jitter_range
            jitter = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = ox + torch.as_tensor(jitter, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._base_half_height + 0.001
        return pos

    def _reset_hang_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_hang_sustain_counter"):
            self._hang_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._hang_sustain_counter[env_idx] = 0

    def _rack_frame_dir(self, local: torch.Tensor) -> torch.Tensor:
        rack_pose = self.rack.pose
        return (rack_pose * Pose.create_from_pq(local)).p - rack_pose.p

    def bar_axis_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        rack_pose = self.rack.pose
        local_mid = torch.zeros((self.num_envs, 3), device=self.device)
        local_mid[:, 2] = self._bar_above_rack_origin
        midpoint = (rack_pose * Pose.create_from_pq(local_mid)).p
        local_dir = torch.zeros((self.num_envs, 3), device=self.device)
        local_dir[:, 1] = 1.0
        direction = self._rack_frame_dir(local_dir)
        return midpoint, direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=1e-9)

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)

        rack_pos = self.rack.pose.p
        saddle_pose = self.saddle.pose
        saddle_pos = saddle_pose.p
        bar_mid, bar_dir = self.bar_axis_world()

        along = ((saddle_pos - bar_mid) * bar_dir).sum(dim=-1)
        on_rail_span = torch.abs(along) <= self._along_bar_tol
        closest = bar_mid + bar_dir * along.unsqueeze(-1)

        local = (saddle_pose.inv() * Pose.create_from_pq(closest)).p
        in_throat = (
            (torch.abs(local[:, 0]) <= self._throat_clear)
            & (local[:, 2] >= self._throat_bottom_local)
            & (local[:, 2] <= self._throat_top_local)
        )

        hangs_below = saddle_pos[:, 2] <= closest[:, 2] - self._hang_below_bar_min

        finger_width = self.gripper_aperture()
        released = (finger_width >= self._released_finger_width) & (
            agent.tcp.pose.p[:, 2] >= closest[:, 2] + self._tcp_clearance_above_bar
        )

        saddle_v = self.saddle.linear_velocity
        supported = (torch.abs(saddle_v[:, 2]) <= self._settled_vz_tol) & (
            torch.linalg.norm(saddle_v - self.rack.linear_velocity, dim=-1) < self._comove_rel_tol
        )

        rack_up = torch.zeros((self.num_envs, 3), device=self.device)
        rack_up[:, 2] = 1.0
        tilt_cos = self._rack_frame_dir(rack_up)[:, 2]
        comove_tol = torch.clamp(
            self._rack_comove_tol_frac * tolerance_speed(self.belt.velocity),
            min=self._rack_comove_tol_floor,
            max=self._rack_comove_tol_ceiling,
        )
        rack_on_belt = (
            (rack_pos[:, 2] > deck_z)
            & (torch.abs(rack_pos[:, 0] - ox) <= self._rack_x_tol)
            & (torch.abs(rack_pos[:, 1] - oy) < self._conveyor_half_length + self._base_half_y)
            & (tilt_cos >= self._rack_tilt_cos)
            & (torch.abs(self.rack.linear_velocity[:, 1] - self.belt.velocity) <= comove_tol)
        )

        hung = in_throat & on_rail_span & hangs_below & released & supported & rack_on_belt

        self._hang_sustain_counter = self._advance_sustain("hang_sustain", self._hang_sustain_counter, hung)
        success = self._hang_sustain_counter >= self._hang_sustain_steps

        return {
            "success": success,
            "in_throat": in_throat,
            "on_rail_span": on_rail_span,
            "hangs_below": hangs_below,
            "released": released,
            "supported": supported,
            "rack_on_belt": rack_on_belt,
            "hang_sustain": self._hang_sustain_counter,
            "saddle_to_rail_cross": torch.abs(local[:, 0]),
            "saddle_along_rail": along,
        }

    def compute_dense_reward(self, obs, action, info):
        saddle_pos = self.saddle.pose.p
        bar_mid, bar_dir = self.bar_axis_world()
        along = ((saddle_pos - bar_mid) * bar_dir).sum(dim=-1)
        closest = bar_mid + bar_dir * along.unsqueeze(-1)

        cross = torch.linalg.norm((saddle_pos - closest)[:, :2], dim=-1)
        align_reward = 1 - torch.tanh(10.0 * cross)
        span_reward = 1 - torch.tanh(5.0 * (torch.abs(along) - self._along_bar_tol).clamp(min=0.0))

        seated = (info["in_throat"] & info["hangs_below"]).float()
        return align_reward + span_reward + 2.0 * seated

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0
