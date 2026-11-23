from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.procedural import build_plate
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_on_belt
from sweep_bench.tasks.conveyor_reach.config import RandomizationConfig
from sweep_bench.tasks.conveyor_reach.scripted import ScriptedReachMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_MARKER_COLOR = (0.90, 0.15, 0.55, 1.0)


@bench_task("ConveyorReach-v0")
@register_env("ConveyorReach-v0", max_episode_steps=300)
class ConveyorReachEnv(ScriptedReachMixin, ConveyorWorkspaceEnv):
    """Reach a moving target point above a marker riding the conveyor belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _marker_radius: float = 0.035
    _marker_half_thickness: float = 0.005
    _marker_patch_radius: float = 0.035

    _target_height: float = 0.12

    _reach_tol: float = 0.03
    _reach_sustain_steps: int = 5

    default_prompt: str = "reach and hold the gripper above the marker on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _marker_origin_above_deck(self) -> float:
        return self._marker_half_thickness + 0.001

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        marker_builder = build_plate(
            self.scene,
            radius=self._marker_radius,
            half_thickness=self._marker_half_thickness,
            color=_MARKER_COLOR,
        )
        marker_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._marker_origin_above_deck])
        self.marker = marker_builder.build(name="reach_marker")
        self.belt.register_rider(self.marker, patch_radius=self._marker_patch_radius)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        marker_pos = self._sample_marker_position(env_idx, b)
        marker_q = torch.zeros((b, 4), device=device)
        marker_q[:, 0] = 1.0
        self.marker.set_pose(Pose.create_from_pq(marker_pos, marker_q))
        marker_vel = torch.zeros((b, 3), device=device)
        marker_vel[:, 1] = self.belt.velocity[env_idx]
        self.marker.set_linear_velocity(marker_vel)
        self.marker.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_reach_trackers(env_idx)

    def _sample_marker_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._marker_radius - 0.04
        marker_pos = torch.zeros((b, 3), device=device)
        marker_pos[:, 0] = ox
        if self.randomization.marker_lateral_offset:
            lane_max = self._conveyor_half_width - self._marker_radius
            xs = self._episode_rng.uniform(-lane_max, lane_max, size=b).astype(np.float32)
            marker_pos[:, 0] = ox + torch.as_tensor(xs, device=device)
        marker_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        marker_pos[:, 2] = deck_z + self._marker_origin_above_deck
        return marker_pos

    def _target_point(self) -> torch.Tensor:
        target = self.marker.pose.p.clone()
        target[:, 2] = target[:, 2] + self._target_height
        return target

    def _reset_reach_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_reach_sustain_counter"):
            self._reach_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reach_sustain_counter[env_idx] = 0

    def evaluate(self):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        target = self._target_point()

        dist = torch.linalg.norm(tcp_pos - target, dim=-1)
        within = dist < self._reach_tol

        self._reach_sustain_counter = self._advance_sustain("reach_sustain", self._reach_sustain_counter, within)
        held = self._reach_sustain_counter >= self._reach_sustain_steps

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        marker_on_belt = rider_on_belt(
            pose=self.marker.pose,
            linear_velocity=self.marker.linear_velocity,
            belt_velocity=self.belt.velocity,
            deck_z=deck_z,
            centre_x=ox,
            centre_y=oy,
            x_tol=self._conveyor_half_width + self._marker_radius,
            y_tol=self._conveyor_half_length + self._marker_radius,
            rest_height=self._marker_half_thickness,
        )

        success = held & marker_on_belt

        return {
            "success": success,
            "within_sphere": within,
            "reach_dist": dist,
            "reach_sustain": self._reach_sustain_counter,
            "marker_on_belt": marker_on_belt,
        }

    def compute_dense_reward(self, obs, action, info):
        return 1.0 - torch.tanh(5.0 * info["reach_dist"])

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)
