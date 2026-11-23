from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply, quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.vetted import resolve_object_specs
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_shoe_mat.assets import build_mat
from sweep_bench.tasks.conveyor_shoe_mat.config import RandomizationConfig
from sweep_bench.tasks.conveyor_shoe_mat.scripted import ScriptedShoeMatMixin
from sweep_bench.tasks.conveyor_shoe_mat.variants import (
    SHOE_MAT_OBJECT_IDS,
    SHOE_SPECS,
    shoe_color_name,
)
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

    from sweep_bench.assets.catalog import ObjectSpec


@bench_task("ConveyorShoeMat-v0")
@register_env("ConveyorShoeMat-v0", max_episode_steps=300)
class ConveyorShoeMatEnv(ScriptedShoeMatMixin, ConveyorWorkspaceEnv):
    """Set the held shoe down SOLE-DOWN on the wall-free mat riding the belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(SHOE_MAT_OBJECT_IDS, extra=SHOE_SPECS)

    _mat_half_x: float = 0.055
    _mat_half_y: float = 0.080
    _mat_half_z: float = 0.007
    _mat_inlay_margin: float = 0.008
    _mat_color: tuple[float, float, float, float] = (0.20, 0.21, 0.24, 1.0)
    _mat_inlay_color: tuple[float, float, float, float] = (0.44, 0.47, 0.52, 1.0)
    _mat_patch_radius: float = 0.055
    _shoe_patch_radius: float = 0.016

    _held_roll_range: tuple[float, float] = (math.radians(28.0), math.radians(36.0))

    _sole_tilt_max_deg: float = 18.0
    _mat_contact_tol: float = 0.005
    _outline_inset: float = 0.005
    _released_finger_width: float = 0.056
    _released_distance_threshold: float = 0.07
    _comove_tol_y: float = 0.05
    _comove_tol_x: float = 0.04
    _mat_x_tol: float = 0.04
    _mat_lift_tol: float = 0.006
    _mat_runway_limit: float = 0.85
    _settle_sustain_steps: int = 10

    default_prompt: str = "set the shoe down flat on its sole on the mat on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _mat_top_local_z(self) -> float:
        return self._mat_half_z

    @property
    def _mat_ride_z(self) -> float:
        return self.workspace_offset[2] + self._conveyor_deck_height + self._mat_half_z

    def mat_frame_points(self, world: torch.Tensor) -> torch.Tensor:
        rot = quaternion_to_matrix(self.mat.pose.q)
        return torch.einsum("bji,bnj->bni", rot, world - self.mat.pose.p.unsqueeze(1))

    def shoe_hull_world(self) -> torch.Tensor:
        pose = self.shoe.pose
        rot = quaternion_to_matrix(pose.q)
        return pose.p.unsqueeze(1) + torch.einsum("bij,bnj->bni", rot, self._shoe_corners_local)

    def sole_normal_world(self) -> torch.Tensor:
        return -quaternion_to_matrix(self.shoe.pose.q)[:, :, 2]

    def sole_down_cos(self) -> torch.Tensor:
        return -self.sole_normal_world()[:, 2]

    def held_drop_offset(self) -> torch.Tensor:
        agent = cast("Panda", self.agent)
        lowest = self.shoe_hull_world()[:, :, 2].min(dim=1).values
        raw = agent.tcp.pose.p[:, 2] - lowest
        return torch.minimum(raw.clamp(min=0.0), self._shoe_hull_radius)

    def _load_objects(self, options: dict) -> None:
        ox, oy, _oz = self.workspace_offset

        mat_builder = build_mat(
            self.scene,
            half_x=self._mat_half_x,
            half_y=self._mat_half_y,
            half_z=self._mat_half_z,
            inlay_margin=self._mat_inlay_margin,
            color=self._mat_color,
            inlay_color=self._mat_inlay_color,
        )
        mat_builder.initial_pose = sapien.Pose(p=[ox, oy, self._mat_ride_z])
        self.mat = mat_builder.build(name="shoe_mat")
        self.belt.register_rider(self.mat, patch_radius=self._mat_patch_radius)

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._shoe_names: list[str] = [str(name) for name in names]
        specs = [spec_by_id[name] for name in self._shoe_names]
        self._shoe_colors: list[str] = [shoe_color_name(spec) for spec in specs]
        self.shoe = self._build_shoe_group(specs)
        self.belt.register_rider(
            self.shoe,
            patch_radius=torch.as_tensor(
                [s.patch_radius if s.patch_radius is not None else self._shoe_patch_radius for s in specs],
                dtype=torch.float32,
                device=self.device,
            ),
        )

        half = torch.as_tensor([s.bbox for s in specs], dtype=torch.float32, device=self.device)
        self._shoe_half = half
        self._shoe_hull_radius = torch.linalg.norm(half, dim=-1)
        self._shoe_grasp_half = torch.as_tensor(
            [s.grasp_width / 2.0 if s.grasp_width is not None else max(s.bbox[0], s.bbox[1]) for s in specs],
            dtype=torch.float32,
            device=self.device,
        )
        signs = torch.tensor(
            [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
            dtype=torch.float32,
            device=self.device,
        )
        self._shoe_corners_local = half.unsqueeze(1) * signs.unsqueeze(0)

    def _build_shoe_group(self, specs: list[ObjectSpec]) -> Actor:
        ox, oy, _oz = self.workspace_offset
        objs: list[Actor] = []
        for i, spec in enumerate(specs):
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, self._mat_ride_z + 0.40 + 0.06 * i])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"shoe-{spec.id}-{i}")
            self.remove_from_state_dict_registry(obj)
            objs.append(obj)
        merged = Actor.merge(objs, name="shoe")
        self.add_to_state_dict_registry(merged)
        return merged

    @property
    def episode_objects(self) -> list[str]:
        return list(self._shoe_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [
            f"set the {color} shoe down flat on its sole on the mat on the conveyor belt" for color in self._shoe_colors
        ]

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        half_w = self._shoe_grasp_half[env_idx].cpu().numpy()
        return 2.0 * (half_w - 0.001)[:, None]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        self._reset_shoe_trackers(env_idx)
        self._held_roll[env_idx] = self._sample_held_roll(b)

        mat_pos = self._sample_mat_position(env_idx, b)
        mat_q = self._sample_mat_orientation(b)
        self.mat.set_pose(Pose.create_from_pq(mat_pos, mat_q))
        mat_vel = torch.zeros((b, 3), device=device)
        mat_vel[:, 1] = self.belt.velocity[env_idx]
        self.mat.set_linear_velocity(mat_vel)
        self.mat.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pose = cast("Panda", self.agent).tcp.pose
        shoe_q = quaternion_multiply(
            tcp_pose.q[env_idx],
            quaternion_multiply(self._roll_quat(self._held_roll[env_idx]), self._grip_frame_q(b)),
        )
        self.shoe.set_pose(Pose.create_from_pq(tcp_pose.p[env_idx], shoe_q))
        self.shoe.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.shoe.set_angular_velocity(torch.zeros((b, 3), device=device))

    def _grip_frame_q(self, b: int) -> torch.Tensor:
        q = torch.zeros((b, 4), device=self.device)
        q[:, 1] = math.sqrt(0.5)
        q[:, 2] = math.sqrt(0.5)
        return q

    def _roll_quat(self, roll: torch.Tensor) -> torch.Tensor:
        q = torch.zeros((roll.shape[0], 4), device=self.device)
        q[:, 0] = torch.cos(0.5 * roll)
        q[:, 2] = torch.sin(0.5 * roll)
        return q

    def held_roll_correction_q(self) -> torch.Tensor:
        return self._roll_quat(-self._held_roll)

    def _sample_held_roll(self, b: int) -> torch.Tensor:
        lo, hi = self._held_roll_range
        mag = self._episode_rng.uniform(lo, hi, size=b)
        sign = self._episode_rng.choice(np.array([-1.0, 1.0]), size=b)
        return torch.as_tensor((mag * sign).astype(np.float32), device=self.device)

    def _sample_mat_orientation(self, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.mat_yaw:
            lo, hi = self.randomization.mat_yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
            q[:, 0] = torch.cos(yaw * 0.5)
            q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_mat_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        device = self.device
        start_y = self._conveyor_half_length - self._mat_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = self._mat_ride_z
        return pos

    def _reset_shoe_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_settle_counter"):
            self._settle_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._held_roll = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._settle_counter[env_idx] = 0

    def evaluate(self):
        agent = cast("Panda", self.agent)
        shoe_pos = self.shoe.pose.p
        mat_pos = self.mat.pose.p
        belt_v = self.belt.velocity

        hull_mat = self.mat_frame_points(self.shoe_hull_world())

        sole_down = self.sole_down_cos() >= math.cos(math.radians(self._sole_tilt_max_deg))

        lowest = hull_mat[:, :, 2].min(dim=1).values
        on_mat_face = torch.abs(lowest - self._mat_top_local_z) <= self._mat_contact_tol

        inside_x = hull_mat[:, :, 0].abs().max(dim=1).values <= self._mat_half_x - self._outline_inset
        inside_y = hull_mat[:, :, 1].abs().max(dim=1).values <= self._mat_half_y - self._outline_inset
        no_overhang = inside_x & inside_y

        finger_width = self.gripper_aperture()
        released = (finger_width >= self._released_finger_width) & (
            torch.linalg.norm(agent.tcp.pose.p - shoe_pos, dim=-1) > self._released_distance_threshold
        )

        shoe_v = self.shoe.linear_velocity
        comoving = (torch.abs(shoe_v[:, 1] - belt_v) < self._comove_tol_y) & (
            torch.abs(shoe_v[:, 0]) < self._comove_tol_x
        )

        ox, oy, _oz = self.workspace_offset
        mat_riding = (
            (torch.abs(mat_pos[:, 0] - ox) <= self._mat_x_tol)
            & (torch.abs(mat_pos[:, 2] - self._mat_ride_z) <= self._mat_lift_tol)
            & (torch.abs(mat_pos[:, 1] - oy) < self._mat_runway_limit)
        )

        seated = sole_down & on_mat_face & no_overhang & released & comoving & mat_riding

        self._settle_counter = self._advance_sustain("shoe_settle", self._settle_counter, seated)
        success = self._settle_counter >= self._settle_sustain_steps

        return {
            "success": success,
            "sole_down": sole_down,
            "on_mat_face": on_mat_face,
            "no_overhang": no_overhang,
            "released": released,
            "comoving": comoving,
            "mat_riding": mat_riding,
            "shoe_settle": self._settle_counter,
            "sole_tilt_deg": torch.rad2deg(torch.arccos(self.sole_down_cos().clamp(-1.0, 1.0))),
            "shoe_to_mat_xy": torch.linalg.norm(shoe_pos[:, :2] - mat_pos[:, :2], dim=-1),
        }

    def compute_dense_reward(self, obs, action, info):
        xy_dist = torch.linalg.norm(self.shoe.pose.p[:, :2] - self.mat.pose.p[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(8 * xy_dist)
        attitude_reward = self.sole_down_cos().clamp(min=0.0)
        seated = (info["sole_down"] & info["on_mat_face"] & info["no_overhang"]).float()
        return carry_reward + attitude_reward + 2.0 * seated

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0
