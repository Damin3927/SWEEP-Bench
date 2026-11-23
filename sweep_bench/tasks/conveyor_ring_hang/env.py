from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.peg_hole import build_hole_block
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_on_belt
from sweep_bench.tasks.conveyor_ring_hang.assets import build_ring_stand
from sweep_bench.tasks.conveyor_ring_hang.config import RandomizationConfig
from sweep_bench.tasks.conveyor_ring_hang.scripted import ScriptedPlaceMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_RING_COLOR = (0.30, 0.45, 0.72, 1.0)


@bench_task("ConveyorRingHang-v0")
@register_env("ConveyorRingHang-v0", max_episode_steps=300)
class ConveyorRingHangEnv(ScriptedPlaceMixin, ConveyorWorkspaceEnv):
    """Hang a held ring onto the vertical post of a stand riding the conveyor belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _base_half_extent: float = 0.05
    _base_half_height: float = 0.01
    _post_radius: float = 0.006
    _post_height: float = 0.05

    _aperture_half_width: float = 0.022
    _ring_half_width: float = 0.030
    _ring_half_height: float = 0.008

    _stand_patch_radius: float = 0.04

    _conveyor_half_width: float = 0.10

    _post_protrusion_min: float = 0.005
    _released_distance_threshold: float = 0.08
    _settled_rel_speed_threshold: float = 0.3
    _hang_sustain_steps: int = 3

    default_prompt: str = "hang the ring on the post on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _aperture_clear(self) -> float:
        return self._aperture_half_width - self._post_radius

    @property
    def _post_top_above_deck(self) -> float:
        return 2.0 * self._base_half_height + self._post_height

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        stand_builder = build_ring_stand(
            self.scene,
            base_half_extent=self._base_half_extent,
            base_half_height=self._base_half_height,
            post_radius=self._post_radius,
            post_height=self._post_height,
        )
        stand_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._base_half_height + 0.001])
        self.stand = stand_builder.build(name="ring_stand")
        self.belt.register_rider(self.stand, patch_radius=self._stand_patch_radius)

        ring_builder = build_hole_block(
            self.scene,
            hole_half_width=self._aperture_half_width,
            block_half_extent=self._ring_half_width,
            block_half_height=self._ring_half_height,
            color=_RING_COLOR,
        )
        ring_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.ring = ring_builder.build(name="ring")

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._ring_half_width - 0.001, dtype=np.float32)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        stand_pos = self._sample_stand_position(env_idx, b)
        stand_q = torch.zeros((b, 4), device=device)
        stand_q[:, 0] = 1.0
        self.stand.set_pose(Pose.create_from_pq(stand_pos, stand_q))
        stand_vel = torch.zeros((b, 3), device=device)
        stand_vel[:, 1] = self.belt.velocity[env_idx]
        self.stand.set_linear_velocity(stand_vel)
        self.stand.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pos = cast("Panda", self.agent).tcp.pose.p[env_idx]
        ring_q = torch.zeros((b, 4), device=device)
        ring_q[:, 0] = 1.0
        self.ring.set_pose(Pose.create_from_pq(tcp_pos, ring_q))
        self.ring.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.ring.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_hang_trackers(env_idx)

    def _sample_stand_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._base_half_extent - 0.04
        stand_pos = torch.zeros((b, 3), device=device)
        stand_pos[:, 0] = ox
        if self.randomization.stand_x_jitter:
            lo, hi = self.randomization.x_jitter_range
            jitter = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            stand_pos[:, 0] = ox + torch.as_tensor(jitter, device=device)
        stand_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        stand_pos[:, 2] = deck_z + self._base_half_height + 0.001
        return stand_pos

    def _reset_hang_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_hang_sustain_counter"):
            self._hang_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._hang_sustain_counter[env_idx] = 0

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        ring_pos = self.ring.pose.p
        stand_pos = self.stand.pose.p
        tcp_pos = cast("Panda", self.agent).tcp.pose.p

        post_top_z = stand_pos[:, 2] + self._base_half_height + self._post_height
        ring_top_z = ring_pos[:, 2] + self._ring_half_height

        xy_dist = torch.linalg.norm(ring_pos[:, :2] - stand_pos[:, :2], dim=-1)
        aperture_aligned = xy_dist < self._aperture_clear
        threaded = aperture_aligned & (ring_top_z < post_top_z - self._post_protrusion_min)

        released = torch.linalg.norm(tcp_pos - ring_pos, dim=-1) > self._released_distance_threshold

        rel_speed = torch.linalg.norm(self.ring.linear_velocity - self.stand.linear_velocity, dim=-1)
        settled = rel_speed < self._settled_rel_speed_threshold

        stand_on_belt = rider_on_belt(
            pose=self.stand.pose,
            linear_velocity=self.stand.linear_velocity,
            belt_velocity=self.belt.velocity,
            deck_z=deck_z,
            centre_x=ox,
            centre_y=oy,
            x_tol=self._conveyor_half_width + self._base_half_extent,
            y_tol=self._conveyor_half_length + self._base_half_extent,
            rest_height=self._base_half_height,
        )

        hung = threaded & released & settled & stand_on_belt
        self._hang_sustain_counter = self._advance_sustain("hang_sustain", self._hang_sustain_counter, hung)
        success = self._hang_sustain_counter >= self._hang_sustain_steps

        return {
            "success": success,
            "threaded": threaded,
            "aperture_aligned": aperture_aligned,
            "released": released,
            "settled": settled,
            "stand_on_belt": stand_on_belt,
            "ring_to_post_xy": xy_dist,
            "hang_sustain": self._hang_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        ring_pos = self.ring.pose.p
        stand_pos = self.stand.pose.p

        xy_dist = torch.linalg.norm(ring_pos[:, :2] - stand_pos[:, :2], dim=-1)
        align_reward = 1 - torch.tanh(10.0 * xy_dist)

        post_top_z = stand_pos[:, 2] + self._base_half_height + self._post_height
        base_top_z = stand_pos[:, 2] + self._base_half_height
        depth = ((post_top_z - self.ring.pose.p[:, 2]) / (post_top_z - base_top_z)).clamp(0.0, 1.0)
        aligned = info["aperture_aligned"].float()
        return align_reward + 2.0 * depth * aligned

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
