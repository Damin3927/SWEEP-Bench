from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.peg_hole import build_hole_block
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import tolerance_speed
from sweep_bench.tasks.conveyor_ring_hang_tilt.assets import build_tilted_ring_stand
from sweep_bench.tasks.conveyor_ring_hang_tilt.config import RandomizationConfig
from sweep_bench.tasks.conveyor_ring_hang_tilt.scripted import ScriptedRingHangTiltMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor

_RING_COLOR = (0.30, 0.45, 0.72, 1.0)


@bench_task("ConveyorRingHangTilt-v0")
@register_env("ConveyorRingHangTilt-v0", max_episode_steps=300)
class ConveyorRingHangTiltEnv(ScriptedRingHangTiltMixin, ConveyorWorkspaceEnv):
    """Hang a held ring onto the OBLIQUE post of a stand riding the conveyor belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _base_half_extent: float = 0.05
    _base_half_height: float = 0.01
    _stand_rest_z_tol: float = 0.005
    _post_radius: float = 0.006
    _post_height: float = 0.05

    _post_tilt_deg: float = 10.0

    _aperture_half_width: float = 0.022
    _ring_half_width: float = 0.027
    _ring_half_height: float = 0.008

    _stand_patch_radius: float = 0.04

    _conveyor_half_width: float = 0.10

    _post_protrusion_min: float = 0.005
    _released_distance_threshold: float = 0.08
    _settled_rel_speed_threshold: float = 0.10
    _hang_sustain_steps: int = 3

    default_prompt: str = "hang the ring on the tilted post on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _post_tilt_rad(self) -> float:
        return math.radians(self._post_tilt_deg)

    @property
    def _aperture_clear(self) -> float:
        return self._aperture_half_width - self._post_radius

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        stand_builder = build_tilted_ring_stand(
            self.scene,
            base_half_extent=self._base_half_extent,
            base_half_height=self._base_half_height,
            post_radius=self._post_radius,
            post_height=self._post_height,
            tilt_rad=self._post_tilt_rad,
        )
        stand_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._base_half_height + 0.001])
        self.stand = stand_builder.build(name="tilted_ring_stand")
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

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.ring]

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

    def post_axis_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        stand_pose = self.stand.pose
        local_root = torch.zeros((self.num_envs, 3), device=self.device)
        local_root[:, 2] = self._base_half_height
        root = (stand_pose * Pose.create_from_pq(local_root)).p

        tilt = self._post_tilt_rad
        local_dir = torch.zeros((self.num_envs, 3), device=self.device)
        local_dir[:, 0] = math.sin(tilt)
        local_dir[:, 2] = math.cos(tilt)
        direction = (stand_pose * Pose.create_from_pq(local_dir)).p - stand_pose.p
        return root, direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=1e-9)

    def _axis_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        root, axis = self.post_axis_world()
        offset = self.ring.pose.p - root
        along = (offset * axis).sum(dim=-1)
        perp = torch.linalg.norm(offset - along.unsqueeze(-1) * axis, dim=-1)
        return along, perp

    def _aperture_crossing(self) -> torch.Tensor:
        root, axis = self.post_axis_world()
        inv = self.ring.pose.inv()
        local_root = (inv * Pose.create_from_pq(root)).p
        local_dir = (inv * Pose.create_from_pq(root + axis)).p - local_root
        dz = local_dir[:, 2]
        degenerate = dz.abs() < 1e-6
        t = -local_root[:, 2] / torch.where(degenerate, torch.ones_like(dz), dz)
        crossing = local_root[:, :2] + t.unsqueeze(-1) * local_dir[:, :2]
        return torch.where(degenerate.unsqueeze(-1), torch.ones_like(crossing), crossing)

    def _ring_axis_half_extent(self) -> torch.Tensor:
        _root, axis = self.post_axis_world()
        normal = quaternion_to_matrix(self.ring.pose.q)[:, :, 2]
        cos_theta = (normal * axis).sum(dim=-1).abs().clamp(min=0.1)
        return self._ring_half_height / cos_theta

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

        along, perp = self._axis_frame()

        crossing = self._aperture_crossing()
        aperture_aligned = crossing.abs().amax(dim=-1) <= self._aperture_half_width
        axis_half_extent = self._ring_axis_half_extent()
        threaded = (
            aperture_aligned
            & (along > 0.0)
            & (along + axis_half_extent < self._post_height - self._post_protrusion_min)
        )

        released = torch.linalg.norm(tcp_pos - ring_pos, dim=-1) > self._released_distance_threshold

        rel_speed = torch.linalg.norm(self.ring.linear_velocity - self.stand.linear_velocity, dim=-1)
        settled = rel_speed < self._settled_rel_speed_threshold

        stand_slip = torch.abs(self.stand.linear_velocity[:, 1] - self.belt.velocity)
        stand_slip_tol = 0.5 * tolerance_speed(self.belt.velocity)
        stand_on_belt = (
            (torch.abs(stand_pos[:, 2] - (deck_z + self._base_half_height)) <= self._stand_rest_z_tol)
            & (torch.abs(stand_pos[:, 0] - ox) < self._conveyor_half_width + self._base_half_extent)
            & (torch.abs(stand_pos[:, 1] - oy) < self._conveyor_half_length + self._base_half_extent)
            & ((torch.abs(self.belt.velocity) <= 1e-6) | (stand_slip <= stand_slip_tol))
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
            "ring_axis_perp": perp,
            "ring_axis_along": along,
            "aperture_crossing": crossing.abs().amax(dim=-1),
            "hang_sustain": self._hang_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        along, perp = self._axis_frame()

        align_reward = 1 - torch.tanh(10.0 * perp)

        depth = ((self._post_height - along) / self._post_height).clamp(0.0, 1.0)
        aligned = info["aperture_aligned"].float()
        return align_reward + 2.0 * depth * aligned

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
