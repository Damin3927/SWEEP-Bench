from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_apply,
    quaternion_invert,
    quaternion_to_matrix,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_handle_grasp.config import (
    HANDLE_YAW_DEFAULT,
    HANDLE_YAW_WIDE,
    RandomizationConfig,
)
from sweep_bench.tasks.conveyor_handle_grasp.scripted import ScriptedHandleGraspMixin
from sweep_bench.tasks.conveyor_handle_grasp.variants import (
    HANDLE_GRASP_OBJECT_IDS,
    HANDLE_VESSEL_BY_ID,
    HANDLE_VESSEL_SPECS,
    HandleVessel,
)
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


def _hull_extreme_points(vessel: HandleVessel) -> np.ndarray:
    pts: list[tuple[float, float, float]] = []
    for k in range(12):
        theta = k * math.pi / 6.0
        x, y = vessel.body_radius * math.cos(theta), vessel.body_radius * math.sin(theta)
        pts.append((x, y, -vessel.body_half_height))
        pts.append((x, y, vessel.body_half_height))
    for sign in (1.0, -1.0):
        pts.append((0.0, sign * vessel.body_radius, vessel.bar_local_z - vessel.bar_radius))
        pts.append((0.0, sign * vessel.body_radius, vessel.bar_local_z + vessel.bar_radius))
    return np.array(pts, dtype=np.float32)


@bench_task("ConveyorHandleGrasp-v0")
@register_env("ConveyorHandleGrasp-v0", max_episode_steps=300)
class ConveyorHandleGraspEnv(ScriptedHandleGraspMixin, ConveyorWorkspaceEnv):
    """Lift a handled vessel off the belt BY ITS HANDLE, not by its body."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(HANDLE_GRASP_OBJECT_IDS, extra=HANDLE_VESSEL_SPECS)

    _patch_radius: float = 0.038

    _grip_below_bar: float = 0.006

    _aperture_squeeze: float = 0.009
    _aperture_slack: float = 0.003
    _handle_box_cross: float = 0.018
    _handle_box_along: float = 0.020
    _handle_box_vert: float = 0.020
    _lift_clearance: float = 0.12
    _carry_speed_match_threshold: float = 0.04
    _vessel_static_speed_threshold: float = 0.10
    _upright_max_tilt: float = math.radians(25.0)
    _on_belt_x_guard: float = 0.11
    _grasp_sustain_steps: int = 8

    _spawn_end_margin: float = 0.022

    default_prompt: str = "pick up the pail by its handle"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}

        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._vessel_names: list[str] = [str(name) for name in names]
        self._vessel_display: list[str] = [display_name(spec_by_id[n]) for n in self._vessel_names]
        vessels = [HANDLE_VESSEL_BY_ID[n] for n in self._vessel_names]

        objs: list[Actor] = []
        patch_radii: list[float] = []
        for i, name in enumerate(self._vessel_names):
            spec = spec_by_id[name]
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy - 0.3, oz + self._conveyor_deck_height + 0.05])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"{name}-{i}")
            self.remove_from_state_dict_registry(obj)
            obj.angular_damping = 5.0
            objs.append(obj)
            patch_radii.append(spec.patch_radius if spec.patch_radius is not None else self._patch_radius)
        self.vessel = Actor.merge(objs, name="vessel")
        self.add_to_state_dict_registry(self.vessel)
        self.belt.register_rider(
            self.vessel,
            patch_radius=torch.as_tensor(patch_radii, dtype=torch.float32, device=self.device),
        )

        def _tensor(values: list[float]) -> torch.Tensor:
            return torch.as_tensor(values, dtype=torch.float32, device=self.device)

        self._body_half_z = _tensor([v.body_half_height for v in vessels])
        self._body_radius = _tensor([v.body_radius for v in vessels])
        self._bar_local_z = _tensor([v.bar_local_z for v in vessels])
        self._grasp_width = _tensor([v.grasp_width for v in vessels])
        pts = [torch.as_tensor(_hull_extreme_points(v), device=self.device) for v in vessels]
        n_max = max(p.shape[0] for p in pts)
        self._hull_pts = torch.stack([torch.cat([p, p[-1:].expand(n_max - p.shape[0], 3)]) for p in pts])

    @property
    def episode_objects(self) -> list[str]:
        return list(self._vessel_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"pick up the {name} by its handle" for name in self._vessel_display]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        oz = self.workspace_offset[2]
        deck_z = oz + self._conveyor_deck_height

        half_range = HANDLE_YAW_WIDE if self.randomization.handle_yaw else HANDLE_YAW_DEFAULT
        yaw = self._episode_rng.uniform(-half_range, half_range, size=b).astype(np.float32)
        yaw_t = torch.as_tensor(yaw, device=device)
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = torch.cos(yaw_t * 0.5)
        q[:, 3] = torch.sin(yaw_t * 0.5)

        pos = self._sample_vessel_position(env_idx, b)
        pos[:, 2] = deck_z + self._body_half_z[env_idx] + 0.001
        self.vessel.set_pose(Pose.create_from_pq(pos, q))

        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        self.vessel.set_linear_velocity(vel)
        self.vessel.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_handle_trackers(env_idx)

    def _sample_vessel_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        device = self.device
        upstream = self._conveyor_half_length - self._body_radius[env_idx] - self._spawn_end_margin
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - upstream * self.belt.direction_sign[env_idx]
        return pos

    def handle_grip_world_pos(self) -> torch.Tensor:
        offset = torch.zeros((self.num_envs, 3), device=self.device)
        offset[:, 2] = self._bar_local_z - self._grip_below_bar
        pose = self.vessel.pose
        return pose.p + quaternion_apply(pose.q, offset)

    def handle_axis_xy(self) -> torch.Tensor:
        local_y = torch.zeros((self.num_envs, 3), device=self.device)
        local_y[:, 1] = 1.0
        axis = quaternion_apply(self.vessel.pose.q, local_y)[:, :2]
        norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
        return torch.where(norm > 1e-3, axis / norm.clamp(min=1e-9), self._scripted_motion_dir)

    def handle_vessel_linear_velocity(self) -> torch.Tensor:
        return self.vessel.linear_velocity

    def _reset_handle_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_handle_sustain_counter"):
            self._handle_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._prev_vessel_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._prev_tcp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._handle_sustain_counter[env_idx] = 0
        self._prev_vessel_pos[env_idx] = self.vessel.pose.p[env_idx]
        self._prev_tcp_pos[env_idx] = cast("Panda", self.agent).tcp.pose.p[env_idx]

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        vessel_pos = self.vessel.pose.p
        vessel_q = self.vessel.pose.q

        vessel_vel = (vessel_pos - self._prev_vessel_pos) * self.control_freq
        rel_vel = ((vessel_pos - tcp_pos) - (self._prev_vessel_pos - self._prev_tcp_pos)) * self.control_freq
        self._prev_vessel_pos = vessel_pos.clone()
        self._prev_tcp_pos = tcp_pos.clone()

        finger_width = self.gripper_aperture()
        rod_aperture = (finger_width > self._grasp_width - self._aperture_squeeze) & (
            finger_width < self._grasp_width + self._aperture_slack
        )
        fingers_loaded = agent.is_grasping(self.vessel)

        tcp_local = quaternion_apply(quaternion_invert(vessel_q), tcp_pos - vessel_pos)
        on_handle = (
            (torch.abs(tcp_local[:, 0]) < self._handle_box_cross)
            & (torch.abs(tcp_local[:, 1]) < self._handle_box_along)
            & (torch.abs(tcp_local[:, 2] - self._bar_local_z) < self._handle_box_vert)
        )

        rot = quaternion_to_matrix(vessel_q)
        lowest_z = vessel_pos[:, 2] + torch.einsum("bj,bvj->bv", rot[:, 2, :], self._hull_pts).amin(dim=-1)
        is_lifted = lowest_z > deck_z + self._lift_clearance

        is_carried = torch.linalg.norm(rel_vel, dim=-1) < self._carry_speed_match_threshold
        is_static = torch.linalg.norm(vessel_vel, dim=-1) < self._vessel_static_speed_threshold

        is_upright = rot[:, 2, 2] > math.cos(self._upright_max_tilt)

        on_belt = (torch.abs(vessel_pos[:, 1] - oy) < self._conveyor_half_length) & (
            torch.abs(vessel_pos[:, 0] - ox) < self._on_belt_x_guard
        )

        holding = rod_aperture & fingers_loaded & on_handle & is_lifted & is_carried & is_static & is_upright & on_belt
        self._handle_sustain_counter = self._advance_sustain("handle_sustain", self._handle_sustain_counter, holding)

        return {
            "success": self._handle_sustain_counter >= self._grasp_sustain_steps,
            "rod_aperture": rod_aperture,
            "fingers_loaded": fingers_loaded,
            "on_handle": on_handle,
            "is_lifted": is_lifted,
            "is_carried": is_carried,
            "is_static": is_static,
            "is_upright": is_upright,
            "on_belt": on_belt,
            "handle_sustain": self._handle_sustain_counter,
            "finger_width": finger_width,
            "vessel_lowest_z": lowest_z,
        }

    def compute_dense_reward(self, obs, action, info):
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        tcp_pos = cast("Panda", self.agent).tcp.pose.p

        tcp_to_grip = torch.linalg.norm(tcp_pos - self.handle_grip_world_pos(), dim=-1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_grip)

        lift_reward = torch.clamp((info["vessel_lowest_z"] - deck_z) / self._lift_clearance, 0, 1)
        return reaching_reward + 2.0 * lift_reward * (info["on_handle"] & info["rod_aperture"]).float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
