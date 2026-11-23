from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_multiply,
    quaternion_to_matrix,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_scoop_ball.assets import build_scoop, scoop_chamfer_drop
from sweep_bench.tasks.conveyor_scoop_ball.config import RandomizationConfig
from sweep_bench.tasks.conveyor_scoop_ball.scripted import ScriptedScoopMixin
from sweep_bench.tasks.conveyor_scoop_ball.variants import BALL_SPECS, SCOOP_BALL_OBJECT_IDS
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorScoopBall-v0")
@register_env("ConveyorScoopBall-v0", max_episode_steps=300)
class ConveyorScoopBallEnv(ScriptedScoopMixin, ConveyorWorkspaceEnv):
    """Catch a rolling ball in a held scoop and lift it clear of the belt."""

    _EARLY_FAIL_HELD_REACH = True

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(SCOOP_BALL_OBJECT_IDS, extra=BALL_SPECS)

    conveyor_speed: float = 0.10

    _scoop_inner_half_x: float = 0.045
    _scoop_inner_half_y: float = 0.030
    _scoop_wall_thickness: float = 0.005
    _scoop_wall_height: float = 0.045
    _scoop_floor_half_thickness: float = 0.0005
    _scoop_chamfer_angle: float = 0.14
    _scoop_stub_half_xy: float = 0.012
    _scoop_stub_half_z: float = 0.020
    _scoop_grip_margin: float = 0.0005

    _scoop_stage_offset: float = 0.08

    _ball_seat_tol: float = 0.004
    _scoop_lift_height: float = 0.10
    _ball_comove_speed: float = 0.08
    _ball_carry_drift: float = 0.012
    _scoop_sustain_steps: int = 12
    _scoop_held_distance: float = 0.05
    _scoop_grip_upper_slack: float = 0.004
    _scoop_grip_lower_slack: float = 0.008
    _ball_off_belt_height: float = 0.05

    default_prompt: str = "scoop the rolling ball off the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _scoop_chamfer_length(self) -> float:
        return 2.0 * self._scoop_floor_half_thickness / math.sin(self._scoop_chamfer_angle)

    @property
    def _scoop_tcp_above_floor(self) -> float:
        return self._scoop_wall_height + self._scoop_stub_half_z

    @property
    def _scoop_below_tcp(self) -> float:
        return self._scoop_tcp_above_floor + scoop_chamfer_drop(
            self._scoop_chamfer_length, self._scoop_chamfer_angle, 0.7 * self._scoop_floor_half_thickness
        )

    @property
    def _scoop_grip_lead(self) -> float:
        return self._scoop_inner_half_y + self._scoop_wall_thickness

    @property
    def _scoop_stub_local(self) -> list[float]:
        return [0.0, self._scoop_grip_lead, self._scoop_tcp_above_floor]

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        names = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._ball_names: list[str] = [str(name) for name in names]
        self._ball_display: list[str] = [display_name(spec_by_id[name]) for name in self._ball_names]
        ball_objs: list[Actor] = []
        radii: list[float] = []
        patch_radii: list[float] = []
        for i, name in enumerate(self._ball_names):
            spec = spec_by_id[name]
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + spec.bbox[2] + 0.001])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"{name}-{i}")
            self.remove_from_state_dict_registry(obj)
            ball_objs.append(obj)
            radii.append(spec.bbox[2])
            patch_radii.append(spec.patch_radius if spec.patch_radius is not None else spec.bbox[2])
        self.ball = Actor.merge(ball_objs, name="scoop_ball")
        self.add_to_state_dict_registry(self.ball)
        self.belt.register_rider(
            self.ball,
            patch_radius=torch.as_tensor(patch_radii, dtype=torch.float32, device=self.device),
        )
        self._scoop_ball_radius = torch.as_tensor(radii, dtype=torch.float32, device=self.device)

        half_xs = self._sample_scoop_half_widths()
        scoop_objs: list[Actor] = []
        for i, half_x in enumerate(half_xs):
            builder = build_scoop(
                self.scene,
                inner_half_x=half_x,
                inner_half_y=self._scoop_inner_half_y,
                wall_thickness=self._scoop_wall_thickness,
                wall_height=self._scoop_wall_height,
                floor_half_thickness=self._scoop_floor_half_thickness,
                chamfer_length=self._scoop_chamfer_length,
                chamfer_angle=self._scoop_chamfer_angle,
                stub_half_xy=self._scoop_stub_half_xy,
                stub_half_height=self._scoop_stub_half_z,
            )
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"scoop-{i}")
            self.remove_from_state_dict_registry(obj)
            scoop_objs.append(obj)
        self.scoop = Actor.merge(scoop_objs, name="scoop")
        self.add_to_state_dict_registry(self.scoop)
        self._scoop_half_x = torch.as_tensor(half_xs, dtype=torch.float32, device=self.device)

        stub = torch.zeros((self.num_envs, 3), device=self.device)
        stub[:, 1] = self._scoop_grip_lead
        stub[:, 2] = self._scoop_tcp_above_floor
        self._scoop_grip_offset = Pose.create_from_pq(p=stub)

    def _sample_scoop_half_widths(self) -> list[float]:
        if not self.randomization.scoop_width:
            return [self._scoop_inner_half_x] * self.num_envs
        lo, hi = self.randomization.scoop_inner_half_x_range
        return [float(v) for v in self._episode_rng.uniform(lo, hi, size=self.num_envs)]

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._scoop_stub_half_xy - self._scoop_grip_margin, dtype=np.float32)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        self._sample_capture_leads(env_idx, b)

        ball_pos = self._sample_ball_position(env_idx, b)
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        self.ball.set_pose(Pose.create_from_pq(ball_pos, q))
        ball_vel = torch.zeros((b, 3), device=device)
        ball_vel[:, 1] = self.belt.velocity[env_idx]
        self.ball.set_linear_velocity(ball_vel)
        self.ball.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pose = cast("Panda", self.agent).tcp.pose
        scoop_q = self._scoop_quat_in_gripper(tcp_pose.q[env_idx], b)
        stub_local = torch.as_tensor(self._scoop_stub_local, dtype=torch.float32, device=device).expand(b, 3)
        stub_world = torch.einsum("bij,bj->bi", quaternion_to_matrix(scoop_q), stub_local)
        self.scoop.set_pose(Pose.create_from_pq(tcp_pose.p[env_idx] - stub_world, scoop_q))
        self.scoop.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.scoop.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_scoop_trackers(env_idx)

    def _scoop_quat_in_gripper(self, tcp_q: torch.Tensor, b: int) -> torch.Tensor:
        offset = torch.zeros((b, 4), device=self.device)
        offset[:, 1] = np.sqrt(0.5)
        offset[:, 2] = np.sqrt(0.5)
        return quaternion_multiply(tcp_q, offset)

    def _sample_capture_leads(self, env_idx: torch.Tensor, b: int) -> None:
        if not hasattr(self, "_scoop_capture_lead"):
            self._scoop_capture_lead = torch.full((self.num_envs,), self._scoop_stage_offset, device=self.device)
        if self.randomization.capture_lead:
            lo, hi = self.randomization.capture_lead_range
            leads = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            self._scoop_capture_lead[env_idx] = torch.as_tensor(leads, device=self.device)
        else:
            self._scoop_capture_lead[env_idx] = self._scoop_stage_offset

    def _sample_ball_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        radius = self._scoop_ball_radius[env_idx]
        start_y = self._conveyor_half_length - radius - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.ball_lateral_offset:
            dx = self._episode_rng.uniform(*self.randomization.ball_x_range, size=b).astype(np.float32)
            pos[:, 0] = pos[:, 0] + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + radius + 0.001
        return pos

    @property
    def episode_objects(self) -> list[str]:
        return list(self._ball_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"scoop the rolling {name} off the conveyor belt" for name in self._ball_display]

    def _ball_local(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.scoop.pose.q)
        return torch.einsum("bji,bj->bi", rot, self.ball.pose.p - self.scoop.pose.p)

    def _ball_in_pocket(self) -> torch.Tensor:
        local = self._ball_local()
        radius = self._scoop_ball_radius
        inside = (torch.abs(local[:, 0]) < self._scoop_half_x) & (torch.abs(local[:, 1]) < self._scoop_inner_half_y)
        seated = torch.abs(local[:, 2] - radius) < self._ball_seat_tol
        return inside & seated

    def _scoop_is_held(self) -> torch.Tensor:
        agent = cast("Panda", self.agent)
        grip_point = (self.scoop.pose * self._scoop_grip_offset).p
        near = torch.linalg.norm(grip_point - agent.tcp.pose.p, dim=-1) < self._scoop_held_distance
        separation = self.gripper_aperture()
        grip_width = 2.0 * self._scoop_stub_half_xy
        spread = (separation < grip_width + self._scoop_grip_upper_slack) & (
            separation > grip_width - self._scoop_grip_lower_slack
        )
        return near & spread

    def _reset_scoop_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_scoop_counter"):
            self._scoop_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._ball_prev_local = self._ball_local()
            self._ball_anchor_local = self._ball_local()
        self._scoop_counter[env_idx] = 0
        self._ball_prev_local[env_idx] = self._ball_local()[env_idx]
        self._ball_anchor_local[env_idx] = self._ball_local()[env_idx]

    def _ball_relative_speed(self) -> torch.Tensor:
        local = self._ball_local()
        speed = torch.linalg.norm(local - self._ball_prev_local, dim=-1) * self.control_freq
        self._ball_prev_local = local
        return speed

    def evaluate(self):
        oz = self.workspace_offset[2]
        deck_z = oz + self._conveyor_deck_height

        in_pocket = self._ball_in_pocket()
        lifted = self.scoop.pose.p[:, 2] > deck_z + self._scoop_lift_height
        relative_speed = self._ball_relative_speed()
        comoving = relative_speed < self._ball_comove_speed
        held = self._scoop_is_held()

        carried = in_pocket & lifted & comoving & held

        local = self._ball_prev_local
        opening = carried & (self._scoop_counter == 0)
        self._ball_anchor_local = torch.where(opening.unsqueeze(-1), local, self._ball_anchor_local)
        drift = torch.linalg.norm(local - self._ball_anchor_local, dim=-1)
        holding = carried & (drift < self._ball_carry_drift)
        self._scoop_counter = self._advance_sustain("scoop", self._scoop_counter, holding)
        success = self._scoop_counter >= self._scoop_sustain_steps

        return {
            "success": success,
            "carried": carried,
            "ball_in_pocket": in_pocket,
            "scoop_lifted": lifted,
            "ball_comoving": comoving,
            "scoop_held": held,
            "scoop_sustain": self._scoop_counter,
            "ball_relative_speed": relative_speed,
            "ball_window_drift": drift,
            "ball_off_belt": self.ball.pose.p[:, 2] > deck_z + self._ball_off_belt_height,
        }

    def compute_dense_reward(self, obs, action, info):
        capture_dist = torch.linalg.norm(self.ball.pose.p - self.scoop.pose.p, dim=-1)
        capture_reward = 1 - torch.tanh(5 * capture_dist)
        carry_reward = (info["scoop_sustain"].float() / self._scoop_sustain_steps).clamp(0.0, 1.0)
        return capture_reward + 2.0 * carry_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
