from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.procedural import build_bin, build_box
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_on_belt
from sweep_bench.tasks.conveyor_orient_place.config import RandomizationConfig
from sweep_bench.tasks.conveyor_orient_place.scripted import ScriptedOrientPlaceMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_BLOCK_COLOR = (0.87, 0.24, 0.20, 1.0)
_TRAY_COLOR = (0.42, 0.44, 0.48, 1.0)


@bench_task("ConveyorOrientPlace-v0")
@register_env("ConveyorOrientPlace-v0", max_episode_steps=300)
class ConveyorOrientPlaceEnv(ScriptedOrientPlaceMixin, ConveyorWorkspaceEnv):
    """Place a held block into a yaw-keyed rectangular slot riding the belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _block_half_x: float = 0.028
    _block_half_y: float = 0.016
    _block_half_z: float = 0.006

    _slot_clearance: float = 0.005
    _slot_inner_half_x: float = _block_half_y + _slot_clearance
    _slot_inner_half_y: float = _block_half_x + _slot_clearance
    _slot_wall_thickness: float = 0.006
    _slot_wall_half_height: float = 0.004
    _slot_floor_half_height: float = 0.004

    _tray_outer_half_x: float = _slot_inner_half_x + 2 * _slot_wall_thickness
    _tray_outer_half_y: float = _slot_inner_half_y + 2 * _slot_wall_thickness
    _tray_patch_radius: float = min(_tray_outer_half_x, _tray_outer_half_y)

    _tray_origin_above_deck: float = _slot_floor_half_height

    _seat_z_tray: float = _slot_floor_half_height + _block_half_z
    _seat_depth_tol: float = 0.004
    _yaw_tol: float = np.deg2rad(12.0)
    _released_distance_threshold: float = 0.08
    _released_finger_width: float = 0.035
    _settled_rel_speed_threshold: float = 0.3
    _place_sustain_steps: int = 5

    default_prompt: str = "place the block into the matching slot on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        tray_builder = build_bin(
            self.scene,
            inner_half_x=self._slot_inner_half_x,
            inner_half_y=self._slot_inner_half_y,
            wall_thickness=self._slot_wall_thickness,
            wall_half_height=self._slot_wall_half_height,
            floor_half_height=self._slot_floor_half_height,
            color=_TRAY_COLOR,
        )
        tray_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._tray_origin_above_deck])
        self.tray = tray_builder.build(name="slot_tray")
        self.belt.register_rider(self.tray, patch_radius=self._tray_patch_radius)

        block_builder = build_box(
            self.scene,
            half_x=self._block_half_x,
            half_y=self._block_half_y,
            half_z=self._block_half_z,
            color=_BLOCK_COLOR,
        )
        block_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.block = block_builder.build(name="place_block")

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._block_half_y - 0.001, dtype=np.float32)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        yaw = self._sample_slot_yaw(env_idx, b)
        if not hasattr(self, "_scripted_slot_yaw"):
            self._scripted_slot_yaw = torch.zeros(self.num_envs, device=device, dtype=torch.float32)
        self._scripted_slot_yaw[env_idx] = yaw
        tray_q = torch.zeros((b, 4), device=device)
        tray_q[:, 0] = torch.cos(0.5 * yaw)
        tray_q[:, 3] = torch.sin(0.5 * yaw)

        tray_pos = self._sample_tray_position(env_idx, b)
        self.tray.set_pose(Pose.create_from_pq(tray_pos, tray_q))
        tray_vel = torch.zeros((b, 3), device=device)
        tray_vel[:, 1] = self.belt.velocity[env_idx]
        self.tray.set_linear_velocity(tray_vel)
        self.tray.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pose = cast("Panda", self.agent).tcp.pose
        block_pos = tcp_pose.p[env_idx]
        block_q = tcp_pose.q[env_idx]
        self.block.set_pose(Pose.create_from_pq(block_pos, block_q))
        self.block.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.block.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_place_trackers(env_idx)

    def _reset_place_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_place_sustain_counter"):
            self._place_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._place_sustain_counter[env_idx] = 0

    def _sample_slot_yaw(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        if not self.randomization.slot_yaw:
            return torch.zeros(b, device=self.device, dtype=torch.float32)
        lo, hi = self.randomization.yaw_range
        yaw = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        return torch.as_tensor(yaw, device=self.device)

    def _sample_tray_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._tray_outer_half_y - 0.04
        tray_pos = torch.zeros((b, 3), device=device)
        tray_pos[:, 0] = ox
        tray_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        tray_pos[:, 2] = deck_z + self._tray_origin_above_deck
        return tray_pos

    def _block_in_slot_frame(self) -> Pose:
        return self.tray.pose.inv() * self.block.pose

    def _yaw_error(self) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

        rb = quaternion_to_matrix(self.block.pose.q)
        rt = quaternion_to_matrix(self.tray.pose.q)
        block_long = rb[:, :2, 0]
        slot_long = rt[:, :2, 1]
        block_long = block_long / torch.linalg.norm(block_long, dim=-1, keepdim=True).clamp(min=1e-9)
        slot_long = slot_long / torch.linalg.norm(slot_long, dim=-1, keepdim=True).clamp(min=1e-9)
        cos = (block_long * slot_long).sum(dim=-1).abs().clamp(0.0, 1.0)
        return torch.arccos(cos)

    def evaluate(self):
        rel = self._block_in_slot_frame().p

        in_slot_xy = (rel[:, 0].abs() < self._slot_inner_half_x) & (rel[:, 1].abs() < self._slot_inner_half_y)
        at_depth = (rel[:, 2] > self._seat_z_tray - self._seat_depth_tol) & (
            rel[:, 2] < self._seat_z_tray + self._seat_depth_tol
        )

        yaw_err = self._yaw_error()
        yaw_ok = yaw_err < self._yaw_tol

        rel_speed = torch.linalg.norm(self.block.linear_velocity - self.tray.linear_velocity, dim=-1)
        settled = rel_speed < self._settled_rel_speed_threshold

        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        released = (self.gripper_aperture() >= self._released_finger_width) & (
            torch.linalg.norm(tcp_pos - self.block.pose.p, dim=-1) > self._released_distance_threshold
        )

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        tray_reach = max(self._tray_outer_half_x, self._tray_outer_half_y)
        tray_on_belt = rider_on_belt(
            pose=self.tray.pose,
            linear_velocity=self.tray.linear_velocity,
            belt_velocity=self.belt.velocity,
            deck_z=deck_z,
            centre_x=ox,
            centre_y=oy,
            x_tol=self._conveyor_half_width + tray_reach,
            y_tol=self._conveyor_half_length + tray_reach,
            rest_height=self._tray_origin_above_deck,
        )

        seated = in_slot_xy & at_depth & yaw_ok & settled & released & tray_on_belt

        self._place_sustain_counter = self._advance_sustain("orient_place", self._place_sustain_counter, seated)
        success = self._place_sustain_counter >= self._place_sustain_steps

        return {
            "success": success,
            "in_slot_xy": in_slot_xy,
            "at_depth": at_depth,
            "yaw_ok": yaw_ok,
            "yaw_error": yaw_err,
            "settled": settled,
            "released": released,
            "tray_on_belt": tray_on_belt,
            "place_sustain": self._place_sustain_counter,
            "block_depth_in_slot": rel[:, 2],
        }

    def compute_dense_reward(self, obs, action, info):
        block_pos = self.block.pose.p
        tray_pos = self.tray.pose.p

        xy_dist = torch.linalg.norm(block_pos[:, :2] - tray_pos[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(5 * xy_dist)

        yaw_reward = (1.0 - info["yaw_error"] / (np.pi / 2)).clamp(0.0, 1.0)

        seated = (info["in_slot_xy"] & info["at_depth"] & info["yaw_ok"]).float()
        return carry_reward + yaw_reward + 2.0 * seated

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0
