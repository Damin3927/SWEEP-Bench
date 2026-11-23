from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_drawer_open.assets import (
    build_drawer_sleeve,
    build_drawer_tray,
)
from sweep_bench.tasks.conveyor_drawer_open.config import RandomizationConfig
from sweep_bench.tasks.conveyor_drawer_open.scripted import ScriptedDrawerOpenMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@bench_task("ConveyorDrawerOpen-v0")
@register_env("ConveyorDrawerOpen-v0", max_episode_steps=300)
class ConveyorDrawerOpenEnv(ScriptedDrawerOpenMixin, ConveyorWorkspaceEnv):
    """Pull a captive drawer tray out to its stop on a sleeve riding the belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _sleeve_half_x: float = 0.040
    _sleeve_half_y: float = 0.045
    _sleeve_height: float = 0.038
    _sleeve_floor_top: float = 0.010
    _sleeve_wall_inner_y: float = 0.041
    _sleeve_lip_inner_y: float = 0.031
    _sleeve_lip_bottom: float = 0.032
    _sleeve_front_stop_x: float = -0.036
    _sleeve_front_stop_inner_y: float = 0.028
    _sleeve_rear_stop_x: float = 0.036
    _sleeve_rear_stop_top: float = 0.024
    _sleeve_engage_dx: float = 0.014
    _sleeve_density: float = 1200.0
    _sleeve_patch_radius: float = 0.030

    _tray_half_x: float = 0.036
    _tray_half_y: float = 0.026
    _tray_thickness: float = 0.020
    _tray_ear_front_x: float = 0.010
    _tray_ear_outer_y: float = 0.037
    _tray_lip_x0: float = -0.040
    _tray_lip_x1: float = -0.030
    _tray_lip_half_y: float = 0.027
    _tray_lip_height: float = 0.022
    _tray_density: float = 320.0
    _tray_friction: float = 0.15

    _drawer_success_stroke: float = 0.030
    _drawer_capture_stroke_max: float = 0.052
    _drawer_capture_dz: float = 0.006
    _drawer_capture_lateral: float = 0.008
    _drawer_withdraw_dist: float = 0.06
    _drawer_belt_half_width: float = 0.08
    _drawer_deck_gap: float = 0.010
    _drawer_sustain_steps: int = 5

    _engage_z_above: float = 0.048
    _wait_z_above: float = 0.075

    default_prompt: str = "slide the drawer tray open on the conveyor belt and let go"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def hard_travel(self) -> float:
        return self._tray_ear_front_x - self._sleeve_front_stop_x

    @property
    def _tray_top(self) -> float:
        return self._sleeve_floor_top + self._tray_thickness

    @property
    def _tray_lip_top(self) -> float:
        return self._tray_top + self._tray_lip_height

    @property
    def _sleeve_origin_above_deck(self) -> float:
        return 0.5 * self._sleeve_height

    @property
    def _tray_origin_above_deck(self) -> float:
        return self._sleeve_floor_top + 0.5 * self._tray_thickness

    @property
    def _tray_rest_dx(self) -> float:
        return -self._sleeve_engage_dx

    @property
    def _tray_rest_dz(self) -> float:
        return self._tray_origin_above_deck - self._sleeve_origin_above_deck

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        sleeve_builder = build_drawer_sleeve(
            self.scene,
            half_x=self._sleeve_half_x,
            half_y=self._sleeve_half_y,
            height=self._sleeve_height,
            floor_top=self._sleeve_floor_top,
            wall_inner_y=self._sleeve_wall_inner_y,
            lip_inner_y=self._sleeve_lip_inner_y,
            lip_bottom=self._sleeve_lip_bottom,
            front_stop_x=self._sleeve_front_stop_x,
            front_stop_inner_y=self._sleeve_front_stop_inner_y,
            rear_stop_x=self._sleeve_rear_stop_x,
            rear_stop_top=self._sleeve_rear_stop_top,
            tray_top=self._tray_top,
            engage_dx=self._sleeve_engage_dx,
            density=self._sleeve_density,
        )
        sleeve_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._sleeve_origin_above_deck])
        self.sleeve = sleeve_builder.build(name="drawer_sleeve")
        self.belt.register_rider(self.sleeve, patch_radius=self._sleeve_patch_radius)

        tray_builder = build_drawer_tray(
            self.scene,
            half_x=self._tray_half_x,
            half_y=self._tray_half_y,
            thickness=self._tray_thickness,
            ear_front_x=self._tray_ear_front_x,
            ear_outer_y=self._tray_ear_outer_y,
            lip_x0=self._tray_lip_x0,
            lip_x1=self._tray_lip_x1,
            lip_half_y=self._tray_lip_half_y,
            lip_height=self._tray_lip_height,
            density=self._tray_density,
            friction=self._tray_friction,
        )
        tray_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._tray_origin_above_deck])
        self.tray = tray_builder.build(name="drawer_tray")

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.tray]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        sleeve_xy = self._sample_sleeve_xy(env_idx, b)
        yaw = self._sample_sleeve_yaw(env_idx, b)
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = torch.cos(0.5 * yaw)
        q[:, 3] = torch.sin(0.5 * yaw)

        offset = self._sleeve_engage_dx
        sleeve_pos = torch.zeros((b, 3), device=device)
        sleeve_pos[:, 0] = sleeve_xy[:, 0] + offset * cos
        sleeve_pos[:, 1] = sleeve_xy[:, 1] + offset * sin
        sleeve_pos[:, 2] = deck_z + self._sleeve_origin_above_deck
        self.sleeve.set_pose(Pose.create_from_pq(sleeve_pos, q))

        start_out = self._sample_start_out(env_idx, b)
        tray_pos = torch.zeros((b, 3), device=device)
        tray_pos[:, 0] = sleeve_xy[:, 0] - start_out * cos
        tray_pos[:, 1] = sleeve_xy[:, 1] - start_out * sin
        tray_pos[:, 2] = deck_z + self._tray_origin_above_deck
        self.tray.set_pose(Pose.create_from_pq(tray_pos, q))

        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        zeros = torch.zeros((b, 3), device=device)
        for actor in (self.sleeve, self.tray):
            actor.set_linear_velocity(vel)
            actor.set_angular_velocity(zeros)

        self._reset_drawer_trackers(env_idx)

    def _sample_sleeve_xy(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        device = self.device
        start_y = self._conveyor_half_length - self._sleeve_half_y - 0.04
        xy = torch.zeros((b, 2), device=device)
        xy[:, 0] = ox
        if self.randomization.rider_lateral_offset:
            lo, hi = self.randomization.lateral_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            xy[:, 0] = xy[:, 0] + torch.as_tensor(dx, device=device)
        xy[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        return xy

    def _sample_sleeve_yaw(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        if not self.randomization.sleeve_yaw:
            return torch.zeros((b,), device=self.device)
        lo, hi = self.randomization.yaw_range
        yaw = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        return torch.as_tensor(yaw, device=self.device)

    def _sample_start_out(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        if not self.randomization.tray_start_out:
            return torch.zeros((b,), device=self.device)
        lo, hi = self.randomization.start_out_range
        frac = self._episode_rng.uniform(lo, hi, size=b)
        return torch.as_tensor((frac * self.hard_travel).astype(np.float32), device=self.device)

    def _sleeve_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rot = quaternion_to_matrix(self.sleeve.pose.q)
        rel_world = self.tray.pose.p - self.sleeve.pose.p
        rel = torch.einsum("bji,bj->bi", rot, rel_world)
        centre_local = torch.zeros((self.num_envs, 3), device=self.device, dtype=rot.dtype)
        centre_local[:, 0] = -self._sleeve_engage_dx
        centre = self.sleeve.pose.p + torch.einsum("bij,bj->bi", rot, centre_local)
        return rel, centre, rot

    def tray_stroke(self) -> torch.Tensor:
        rel, _centre, _rot = self._sleeve_frame()
        return self._tray_rest_dx - rel[:, 0]

    def _reset_drawer_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_drawer_sustain_counter"):
            self._drawer_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._drawer_sustain_counter[env_idx] = 0

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        rel, centre, rot = self._sleeve_frame()

        stroke = self._tray_rest_dx - rel[:, 0]
        drawn = stroke >= self._drawer_success_stroke

        not_free = stroke <= self._drawer_capture_stroke_max
        seated = torch.abs(rel[:, 2] - self._tray_rest_dz) < self._drawer_capture_dz
        centred = torch.abs(rel[:, 1]) < self._drawer_capture_lateral
        captured = not_free & seated & centred

        agent = cast("Panda", self.agent)
        tcp_dist = torch.linalg.norm(agent.tcp.pose.p - self.tray.pose.p, dim=-1)
        withdrawn = tcp_dist >= self._drawer_withdraw_dist

        half = torch.as_tensor(
            [self._sleeve_half_x, self._sleeve_half_y, 0.5 * self._sleeve_height],
            device=self.device,
            dtype=rot.dtype,
        )
        x_in_body = rot[:, 0, :]
        up_in_body = rot[:, 2, :]
        half_x_extent = (half * x_in_body.abs()).sum(dim=-1)
        half_z_extent = (half * up_in_body.abs()).sum(dim=-1)
        on_belt = (
            (torch.abs(centre[:, 0] - ox) + half_x_extent < self._drawer_belt_half_width)
            & (torch.abs(centre[:, 2] - half_z_extent - deck_z) < self._drawer_deck_gap)
            & (torch.abs(centre[:, 1] - oy) < self._conveyor_half_length)
        )

        holding = drawn & captured & withdrawn & on_belt
        self._drawer_sustain_counter = self._advance_sustain("drawer_sustain", self._drawer_sustain_counter, holding)
        success = self._drawer_sustain_counter >= self._drawer_sustain_steps

        return {
            "success": success,
            "tray_stroke": stroke,
            "drawn": drawn,
            "tray_captured": captured,
            "tray_seated": seated,
            "tray_centred": centred,
            "arm_clear": withdrawn,
            "sleeve_on_belt": on_belt,
            "drawer_sustain": self._drawer_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p

        tcp_to_lane = torch.linalg.norm(tcp_pos - self.sleeve.pose.p, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_lane)

        stroke_reward = (info["tray_stroke"] / self._drawer_success_stroke).clamp(0.0, 1.0)

        return reach_reward + 2.0 * stroke_reward + 2.0 * info["success"].to(torch.float32)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
