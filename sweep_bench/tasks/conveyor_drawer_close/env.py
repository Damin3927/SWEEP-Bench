from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_drawer_close.assets import (
    build_drawer_sleeve,
    build_drawer_tray,
)
from sweep_bench.tasks.conveyor_drawer_close.config import RandomizationConfig
from sweep_bench.tasks.conveyor_drawer_close.scripted import ScriptedDrawerCloseMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@bench_task("ConveyorDrawerClose-v0")
@register_env("ConveyorDrawerClose-v0", max_episode_steps=300)
class ConveyorDrawerCloseEnv(ScriptedDrawerCloseMixin, ConveyorWorkspaceEnv):
    """Push a captive sliding tray shut on a carrier riding the conveyor belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _sleeve_half_x: float = 0.050
    _sleeve_half_y: float = 0.035
    _sleeve_channel_half_y: float = 0.030
    _sleeve_channel_height: float = 0.026
    _sleeve_floor_half_thickness: float = 0.004
    _sleeve_top_half_thickness: float = 0.003
    _sleeve_back_half_x: float = 0.005
    _sleeve_density: float = 1000.0
    _sleeve_friction: float = 0.05

    _sleeve_patch_radius: float = 0.03

    _tray_body_half_x: float = 0.040
    _tray_body_half_y: float = 0.027
    _tray_body_half_z: float = 0.012
    _tray_front_half_x: float = 0.004
    _tray_front_half_y: float = 0.035
    _tray_front_half_z: float = 0.015
    _tray_density: float = 400.0

    _tray_start_out: float = 0.028

    _drawer_closed_tol: float = 0.006
    _drawer_capture_dz: float = 0.006
    _drawer_capture_dy: float = 0.008
    _drawer_hands_off: float = 0.06
    _drawer_deck_drop: float = 0.005
    _drawer_sustain_steps: int = 5

    default_prompt: str = "push the drawer shut on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _sleeve_origin_above_deck(self) -> float:
        return 2.0 * self._sleeve_floor_half_thickness

    @property
    def _sleeve_top_local_z(self) -> float:
        return self._sleeve_channel_height + 2.0 * self._sleeve_top_half_thickness

    @property
    def _tray_rest_local_z(self) -> float:
        return self._tray_body_half_z

    @property
    def _tray_closed_local_x(self) -> float:
        return self._tray_body_half_x - self._sleeve_half_x

    @property
    def _tray_front_dz(self) -> float:
        return self._tray_front_half_z - self._tray_body_half_z

    @property
    def _tray_push_local(self) -> tuple[float, float, float]:
        return (-(self._tray_body_half_x + 2.0 * self._tray_front_half_x), 0.0, 0.006)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        sleeve_builder = build_drawer_sleeve(
            self.scene,
            outer_half_x=self._sleeve_half_x,
            outer_half_y=self._sleeve_half_y,
            channel_half_y=self._sleeve_channel_half_y,
            channel_height=self._sleeve_channel_height,
            floor_half_thickness=self._sleeve_floor_half_thickness,
            top_half_thickness=self._sleeve_top_half_thickness,
            back_half_x=self._sleeve_back_half_x,
            density=self._sleeve_density,
            friction=self._sleeve_friction,
        )
        sleeve_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._sleeve_origin_above_deck])
        self.sleeve = sleeve_builder.build(name="drawer_sleeve")
        self.belt.register_rider(self.sleeve, patch_radius=self._sleeve_patch_radius)

        tray_builder = build_drawer_tray(
            self.scene,
            body_half_x=self._tray_body_half_x,
            body_half_y=self._tray_body_half_y,
            body_half_z=self._tray_body_half_z,
            front_half_x=self._tray_front_half_x,
            front_half_y=self._tray_front_half_y,
            front_half_z=self._tray_front_half_z,
            front_dz=self._tray_front_dz,
            density=self._tray_density,
        )
        tray_builder.initial_pose = sapien.Pose(
            p=[ox, oy, deck_z + self._sleeve_origin_above_deck + self._tray_rest_local_z]
        )
        self.tray = tray_builder.build(name="drawer_tray")

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.tray]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        sleeve_q = self._sample_sleeve_orientation(env_idx, b)
        sleeve_pos = self._sample_sleeve_position(env_idx, b)
        self.sleeve.set_pose(Pose.create_from_pq(sleeve_pos, sleeve_q))
        sleeve_vel = torch.zeros((b, 3), device=device)
        sleeve_vel[:, 1] = self.belt.velocity[env_idx]
        self.sleeve.set_linear_velocity(sleeve_vel)
        self.sleeve.set_angular_velocity(torch.zeros((b, 3), device=device))

        start_out = self._sample_tray_start_out(env_idx, b)
        local = torch.zeros((b, 3), device=device)
        local[:, 0] = self._tray_closed_local_x - start_out
        local[:, 2] = self._tray_rest_local_z
        rot = quaternion_to_matrix(sleeve_q)
        tray_pos = sleeve_pos + torch.bmm(rot, local.unsqueeze(-1)).squeeze(-1)
        self.tray.set_pose(Pose.create_from_pq(tray_pos, sleeve_q))
        tray_vel = torch.zeros((b, 3), device=device)
        tray_vel[:, 1] = self.belt.velocity[env_idx]
        self.tray.set_linear_velocity(tray_vel)
        self.tray.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_drawer_trackers(env_idx)

    def _sample_sleeve_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.sleeve_yaw:
            lo, hi = self.randomization.sleeve_yaw_range
            yaw = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            yaw_t = torch.as_tensor(yaw, device=device)
            q[:, 0] = torch.cos(yaw_t * 0.5)
            q[:, 3] = torch.sin(yaw_t * 0.5)
        return q

    def _sample_tray_start_out(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        if not self.randomization.tray_start_out:
            return torch.full((b,), self._tray_start_out, device=device)
        lo, hi = self.randomization.tray_start_out_range
        return torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)

    def _sample_sleeve_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._sleeve_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._sleeve_origin_above_deck
        return pos

    def _tray_carrier_offset(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.sleeve.pose.q)
        rel_world = (self.tray.pose.p - self.sleeve.pose.p).unsqueeze(-1)
        return torch.bmm(rot.transpose(1, 2), rel_world).squeeze(-1)

    def _tray_stroke_axis(self) -> torch.Tensor:
        return quaternion_to_matrix(self.sleeve.pose.q)[:, :, 0]

    def _tray_push_point(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.tray.pose.q)
        local = torch.as_tensor(self._tray_push_local, device=self.device, dtype=rot.dtype)
        return self.tray.pose.p + (rot @ local)

    def _tray_closure(self) -> torch.Tensor:
        return self._tray_closed_local_x - self._tray_carrier_offset()[:, 0]

    def _sleeve_half_x_extent(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.sleeve.pose.q)
        half = torch.as_tensor(
            [self._sleeve_half_x, self._sleeve_half_y, self._sleeve_top_local_z],
            device=self.device,
            dtype=rot.dtype,
        )
        return (half * rot[:, 0, :].abs()).sum(dim=-1)

    def _reset_drawer_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_drawer_sustain_counter"):
            self._drawer_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._drawer_sustain_counter[env_idx] = 0

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        offset = self._tray_carrier_offset()
        closure = self._tray_closed_local_x - offset[:, 0]
        closed = closure.abs() <= self._drawer_closed_tol

        captured = (torch.abs(offset[:, 2] - self._tray_rest_local_z) < self._drawer_capture_dz) & (
            torch.abs(offset[:, 1]) < self._drawer_capture_dy
        )

        agent = cast("Panda", self.agent)
        tcp_to_face = torch.linalg.norm(agent.tcp.pose.p - self._tray_push_point(), dim=-1)
        hands_off = tcp_to_face >= self._drawer_hands_off

        sleeve_pos = self.sleeve.pose.p
        on_belt = (
            (torch.abs(sleeve_pos[:, 0] - ox) + self._sleeve_half_x_extent() <= self._conveyor_half_width)
            & (sleeve_pos[:, 2] > deck_z + self._sleeve_origin_above_deck - self._drawer_deck_drop)
            & (torch.abs(sleeve_pos[:, 1] - oy) < self._conveyor_half_length)
        )

        holding = closed & captured & hands_off & on_belt
        self._drawer_sustain_counter = self._advance_sustain("drawer_sustain", self._drawer_sustain_counter, holding)
        success = self._drawer_sustain_counter >= self._drawer_sustain_steps

        return {
            "success": success,
            "closed": closed,
            "closure": closure,
            "captured": captured,
            "hands_off": hands_off,
            "tcp_to_face": tcp_to_face,
            "sleeve_on_belt": on_belt,
            "drawer_sustain": self._drawer_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        tcp_to_face = info["tcp_to_face"]

        reach_reward = 1 - torch.tanh(5 * tcp_to_face)

        close_reward = 1.0 - (info["closure"].abs() / self._tray_start_out).clamp(0.0, 1.0)

        withdraw_reward = info["closed"].to(tcp_to_face.dtype) * (tcp_to_face / self._drawer_hands_off).clamp(0.0, 1.0)

        return reach_reward + 2.0 * close_reward + withdraw_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0
