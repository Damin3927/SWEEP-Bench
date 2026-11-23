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

from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_tray_lift.assets import build_ball
from sweep_bench.tasks.conveyor_tray_lift.config import RandomizationConfig
from sweep_bench.tasks.conveyor_tray_lift.scripted import ScriptedTrayLiftMixin
from sweep_bench.tasks.conveyor_tray_lift.variants import (
    TRAY_FLOOR_HALF_HEIGHT,
    TRAY_INNER_HALF_X,
    TRAY_LIFT_OBJECT_IDS,
    TRAY_LIFT_TRAY_SPECS,
    TRAY_WALL_HALF_HEIGHT,
    TRAY_WALL_THICKNESS,
)
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorTrayLift-v0")
@register_env("ConveyorTrayLift-v0", max_episode_steps=300)
class ConveyorTrayLiftEnv(ScriptedTrayLiftMixin, ConveyorWorkspaceEnv):
    """Lift a loaded tray off the belt without spilling its payload."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(TRAY_LIFT_OBJECT_IDS, extra=TRAY_LIFT_TRAY_SPECS)

    _tray_wall_thickness: float = TRAY_WALL_THICKNESS
    _tray_wall_half_height: float = TRAY_WALL_HALF_HEIGHT
    _tray_floor_half_height: float = TRAY_FLOOR_HALF_HEIGHT
    _tray_inner_half_x: float = TRAY_INNER_HALF_X

    _payload_radius: float = 0.010
    _payload_density: float = 2500.0
    _payload_color: tuple[float, float, float, float] = (0.92, 0.50, 0.12, 1.0)
    _payload_angular_damping: float = 5.0

    _spawn_end_margin: float = 0.04

    _patch_radius: float = 0.02

    _lift_clearance: float = 0.12
    _grip_gap_squeeze_max: float = 0.024
    _grip_gap_slack_max: float = 0.006
    _tray_carry_speed_match: float = 0.04
    _payload_wall_slack: float = 0.002
    _payload_floor_gap: float = 0.004
    _payload_speed_match: float = 0.05
    _tray_level_cos: float = math.cos(math.radians(15.0))
    _sustain_steps: int = 8

    default_prompt: str = "lift the tray off the conveyor belt without spilling the ball"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _tray_half_z(self) -> float:
        return self._tray_floor_half_height + self._tray_wall_half_height

    @property
    def _tray_floor_top_local_z(self) -> float:
        return -self._tray_half_z + 2 * self._tray_floor_half_height

    @property
    def _tray_grasp_width(self) -> float:
        return 2 * (self._tray_inner_half_x + 2 * self._tray_wall_thickness)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        drawn = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._tray_names: list[str] = [str(name) for name in drawn]
        self._tray_display: list[str] = [display_name(spec_by_id[n]) for n in self._tray_names]

        trays: list[Actor] = []
        patch_radii: list[float] = []
        for i, name in enumerate(self._tray_names):
            spec = spec_by_id[name]
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + spec.bbox[2] + 0.001])
            builder.set_scene_idxs([i])
            tray = builder.build(name=f"tray-{name}-{i}")
            self.remove_from_state_dict_registry(tray)
            trays.append(tray)
            patch_radii.append(spec.patch_radius if spec.patch_radius is not None else self._patch_radius)
        self.tray = Actor.merge(trays, name="tray")
        self.add_to_state_dict_registry(self.tray)
        self.belt.register_rider(
            self.tray,
            patch_radius=torch.as_tensor(patch_radii, dtype=torch.float32, device=self.device),
        )

        ball_builder = build_ball(
            self.scene,
            radius=self._payload_radius,
            color=self._payload_color,
            density=self._payload_density,
            initial_pose=sapien.Pose(p=[ox, oy, deck_z + 0.1]),
        )
        self.ball = ball_builder.build(name="payload")
        self.ball.angular_damping = self._payload_angular_damping

        def _tensor(values: list[float]) -> torch.Tensor:
            return torch.as_tensor(values, dtype=torch.float32, device=self.device)

        bboxes = [spec_by_id[n].bbox for n in self._tray_names]
        self._tray_bbox_half_z = _tensor([b[2] for b in bboxes])
        self._tray_half_footprint = _tensor([max(b[0], b[1]) for b in bboxes])
        self._tray_inner_half_y = _tensor([b[1] - 2 * self._tray_wall_thickness for b in bboxes])

    @property
    def episode_objects(self) -> list[str]:
        return list(self._tray_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"lift the {name} off the conveyor belt without spilling the ball" for name in self._tray_display]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        oz = self.workspace_offset[2]
        deck_z = oz + self._conveyor_deck_height

        upstream = self._conveyor_half_length - self._tray_half_footprint[env_idx] - self._spawn_end_margin
        center_xy = self._sample_tray_center_xy(env_idx, b, upstream)

        tray_pos = torch.zeros((b, 3), device=device)
        tray_pos[:, 0] = center_xy[:, 0]
        tray_pos[:, 1] = center_xy[:, 1]
        tray_pos[:, 2] = deck_z + self._tray_bbox_half_z[env_idx] + 0.001
        upright = torch.zeros((b, 4), device=device)
        upright[:, 0] = 1.0
        self.tray.set_pose(Pose.create_from_pq(tray_pos, upright))

        ball_pos = tray_pos.clone()
        ball_pos[:, 2] = tray_pos[:, 2] + self._tray_floor_top_local_z + self._payload_radius
        ball_pos[:, :2] = ball_pos[:, :2] + self._sample_payload_offset(env_idx, b)
        self.ball.set_pose(Pose.create_from_pq(ball_pos, upright.clone()))

        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        zero = torch.zeros((b, 3), device=device)
        self.tray.set_linear_velocity(vel)
        self.tray.set_angular_velocity(zero)
        self.ball.set_linear_velocity(vel.clone())
        self.ball.set_angular_velocity(zero.clone())

        self._reset_tray_lift_trackers(env_idx)

    def _sample_payload_offset(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        offset = torch.zeros((b, 2), device=device)
        rcfg = self.randomization
        if not rcfg.payload_offset:
            return offset
        free_x = self._tray_inner_half_x - self._payload_radius
        free_y = self._tray_inner_half_y[env_idx] - self._payload_radius
        u = torch.as_tensor(self._episode_rng.uniform(-1.0, 1.0, size=(b, 2)).astype(np.float32), device=device)
        offset[:, 0] = u[:, 0] * rcfg.payload_offset_frac * free_x
        offset[:, 1] = u[:, 1] * rcfg.payload_offset_frac * free_y
        return offset

    def _sample_tray_center_xy(self, env_idx: torch.Tensor, b: int, upstream: torch.Tensor) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        device = self.device
        xy = torch.zeros((b, 2), device=device)
        xy[:, 0] = ox
        xy[:, 1] = oy - upstream * self.belt.direction_sign[env_idx]
        return xy

    def _reset_tray_lift_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_lift_sustain_counter"):
            self._lift_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._prev_tray_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._prev_ball_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._prev_tcp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._lift_sustain_counter[env_idx] = 0
        self._prev_tray_pos[env_idx] = self.tray.pose.p[env_idx]
        self._prev_ball_pos[env_idx] = self.ball.pose.p[env_idx]
        self._prev_tcp_pos[env_idx] = cast("Panda", self.agent).tcp.pose.p[env_idx]

    def evaluate(self):
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tray_pos = self.tray.pose.p
        ball_pos = self.ball.pose.p
        rot = quaternion_to_matrix(self.tray.pose.q)

        tray_rel_tcp = ((tray_pos - tcp_pos) - (self._prev_tray_pos - self._prev_tcp_pos)) * self.control_freq
        ball_rel_tray = ((ball_pos - tray_pos) - (self._prev_ball_pos - self._prev_tray_pos)) * self.control_freq
        self._prev_tray_pos = tray_pos.clone()
        self._prev_ball_pos = ball_pos.clone()
        self._prev_tcp_pos = tcp_pos.clone()

        floor_top_z = tray_pos[:, 2] + rot[:, 2, 2] * self._tray_floor_top_local_z
        is_lifted = floor_top_z > deck_z + self._lift_clearance

        finger_gap = self.gripper_aperture()
        width = self._tray_grasp_width
        is_gripped = (finger_gap > width - self._grip_gap_squeeze_max) & (finger_gap < width + self._grip_gap_slack_max)

        is_carried = torch.linalg.norm(tray_rel_tcp, dim=-1) < self._tray_carry_speed_match

        local = torch.einsum("bji,bj->bi", rot, ball_pos - tray_pos)
        free_x = self._tray_inner_half_x - self._payload_radius + self._payload_wall_slack
        free_y = self._tray_inner_half_y - self._payload_radius + self._payload_wall_slack
        payload_inside = (torch.abs(local[:, 0]) < free_x) & (torch.abs(local[:, 1]) < free_y)
        payload_seated = (
            torch.abs(local[:, 2] - self._payload_radius - self._tray_floor_top_local_z) < self._payload_floor_gap
        )

        payload_still = torch.linalg.norm(ball_rel_tray, dim=-1) < self._payload_speed_match

        tray_level = rot[:, 2, 2] > self._tray_level_cos

        holding = is_lifted & is_gripped & is_carried & payload_inside & payload_seated & payload_still & tray_level

        self._lift_sustain_counter = self._advance_sustain("lift_sustain", self._lift_sustain_counter, holding)

        return {
            "success": self._lift_sustain_counter >= self._sustain_steps,
            "is_lifted": is_lifted,
            "is_gripped": is_gripped,
            "is_carried": is_carried,
            "payload_inside": payload_inside,
            "payload_seated": payload_seated,
            "payload_still": payload_still,
            "tray_level": tray_level,
            "lift_sustain": self._lift_sustain_counter,
            "finger_gap": finger_gap,
            "tray_floor_z": floor_top_z,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tray_pos = self.tray.pose.p
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        tcp_to_tray = torch.linalg.norm(tcp_pos - tray_pos, dim=-1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_tray)

        lift_reward = torch.clamp((info["tray_floor_z"] - deck_z) / self._lift_clearance, 0, 1)

        retained = (info["payload_inside"] & info["payload_seated"]).float()
        return reaching_reward + 2.0 * lift_reward * info["is_lifted"].float() * retained

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
