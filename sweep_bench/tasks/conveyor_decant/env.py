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

from sweep_bench.assets.procedural import build_bin
from sweep_bench.assets.vetted import display_name, resolve_object_specs
from sweep_bench.tasks.conveyor_base.cameras import human_render_camera_config
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_decant.assets import build_payload_ball
from sweep_bench.tasks.conveyor_decant.config import RandomizationConfig
from sweep_bench.tasks.conveyor_decant.scripted import ScriptedDecantMixin
from sweep_bench.tasks.conveyor_decant.variants import (
    CUP_FLOOR_HALF_HEIGHT,
    CUP_INNER_HALF_X,
    CUP_WALL_HALF_HEIGHT,
    CUP_WALL_THICKNESS,
    DECANT_CUP_SPECS,
    DECANT_OBJECT_IDS,
)
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_BIN_COLOR = (0.32, 0.36, 0.44, 1.0)


@bench_task("ConveyorDecant-v0")
@register_env("ConveyorDecant-v0", max_episode_steps=300)
class ConveyorDecantEnv(ScriptedDecantMixin, ConveyorWorkspaceEnv):
    """Tip the cup out into the bin without ever touching what is inside."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    OBJECT_CATALOG = resolve_object_specs(DECANT_OBJECT_IDS, extra=DECANT_CUP_SPECS)

    _cup_wall_thickness: float = CUP_WALL_THICKNESS
    _cup_wall_half_height: float = CUP_WALL_HALF_HEIGHT
    _cup_floor_half_height: float = CUP_FLOOR_HALF_HEIGHT
    _cup_inner_half_x: float = CUP_INNER_HALF_X

    _payload_radius: float = 0.011
    _payload_density: float = 1000.0
    _payload_color: tuple[float, float, float, float] = (0.92, 0.50, 0.12, 1.0)
    _payload_angular_damping: float = 5.0

    _bin_inner_half_x: float = 0.08
    _bin_inner_half_y: float = 0.08
    _bin_wall_thickness: float = 0.008
    _bin_wall_half_height: float = 0.030

    @property
    def _bin_floor_half_height(self) -> float:
        return self._conveyor_deck_height / 2.0

    _bin_offset_x: float = -0.28
    _bin_offset_y_mag: float = 0.35
    _bin_fast_band_speed: float = 0.25

    _spawn_end_margin: float = 0.04
    _patch_radius: float = 0.02

    _payload_settled_speed: float = 0.05
    _cup_carry_speed_match: float = 0.04
    _cup_hold_dist: float = 0.06
    _grip_gap_squeeze_max: float = 0.020
    _grip_gap_slack_max: float = 0.008
    _payload_box_slack: float = 0.004
    _tip_exit_cos: float = math.cos(math.radians(60.0))
    _ball_touch_force_threshold: float = 0.05
    _sustain_steps: int = 8

    default_prompt: str = "tip the ball out of the cup into the bin without touching the ball"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _cup_half_z(self) -> float:
        return self._cup_floor_half_height + self._cup_wall_half_height

    @property
    def _cup_floor_top_local_z(self) -> float:
        return -self._cup_half_z + 2 * self._cup_floor_half_height

    @property
    def _cup_grasp_width(self) -> float:
        return 2 * (self._cup_inner_half_x + 2 * self._cup_wall_thickness)

    @property
    def _bin_rim_local_z(self) -> float:
        return self._bin_floor_half_height + 2.0 * self._bin_wall_half_height

    @property
    def _bin_origin_above_surface(self) -> float:
        return self._bin_floor_half_height

    _human_render_eye: tuple[float, float, float] = (0.75, 0.0, 1.25)
    _human_render_target: tuple[float, float, float] = (-0.1, 0.0, 0.15)

    @property
    def _default_human_render_camera_configs(self):
        return human_render_camera_config(
            self.workspace_offset,
            self.shader_pack,
            eye=self._human_render_eye,
            target=self._human_render_target,
        )

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        spec_by_id = {spec.id: spec for spec in self.OBJECT_CATALOG}
        drawn = self._batched_episode_rng.choice(np.array(self.objects), replace=True)
        self._cup_names: list[str] = [str(name) for name in drawn]
        self._cup_display: list[str] = [display_name(spec_by_id[n]) for n in self._cup_names]

        cups: list[Actor] = []
        patch_radii: list[float] = []
        for i, name in enumerate(self._cup_names):
            spec = spec_by_id[name]
            builder = spec.builder_fn(self.scene)
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + spec.bbox[2] + 0.001])
            builder.set_scene_idxs([i])
            cup = builder.build(name=f"cup-{name}-{i}")
            self.remove_from_state_dict_registry(cup)
            cups.append(cup)
            patch_radii.append(spec.patch_radius if spec.patch_radius is not None else self._patch_radius)
        self.cup = Actor.merge(cups, name="cup")
        self.add_to_state_dict_registry(self.cup)
        self.belt.register_rider(
            self.cup,
            patch_radius=torch.as_tensor(patch_radii, dtype=torch.float32, device=self.device),
        )

        ball_builder = build_payload_ball(
            self.scene,
            radius=self._payload_radius,
            color=self._payload_color,
            density=self._payload_density,
            initial_pose=sapien.Pose(p=[ox, oy, deck_z + 0.1]),
        )
        self.ball = ball_builder.build(name="payload")
        self.ball.angular_damping = self._payload_angular_damping

        bin_builder = build_bin(
            self.scene,
            inner_half_x=self._bin_inner_half_x,
            inner_half_y=self._bin_inner_half_y,
            wall_thickness=self._bin_wall_thickness,
            wall_half_height=self._bin_wall_half_height,
            floor_half_height=self._bin_floor_half_height,
            color=_BIN_COLOR,
            initial_pose=sapien.Pose(
                p=[ox + self._bin_offset_x, oy + self._bin_offset_y_mag, oz + self._bin_origin_above_surface]
            ),
        )
        self.bin = bin_builder.build_kinematic(name="dropoff_bin")

        def _tensor(values: list[float]) -> torch.Tensor:
            return torch.as_tensor(values, dtype=torch.float32, device=self.device)

        bboxes = [spec_by_id[n].bbox for n in self._cup_names]
        self._cup_bbox_half_z = _tensor([b[2] for b in bboxes])
        self._cup_half_footprint = _tensor([max(b[0], b[1]) for b in bboxes])
        self._cup_inner_half_y = _tensor([b[1] - 2 * self._cup_wall_thickness for b in bboxes])

    @property
    def episode_objects(self) -> list[str]:
        return list(self._cup_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"tip the ball out of the {name} into the bin without touching the ball" for name in self._cup_display]

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.bin]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        oz = self.workspace_offset[2]
        deck_z = oz + self._conveyor_deck_height

        bin_pos = self._sample_bin_position(env_idx, b)
        upright = torch.zeros((b, 4), device=device)
        upright[:, 0] = 1.0
        self.bin.set_pose(Pose.create_from_pq(bin_pos, upright.clone()))

        upstream = self._conveyor_half_length - self._cup_half_footprint[env_idx] - self._spawn_end_margin
        center_xy = self._sample_cup_center_xy(env_idx, b, upstream)
        cup_pos = torch.zeros((b, 3), device=device)
        cup_pos[:, 0] = center_xy[:, 0]
        cup_pos[:, 1] = center_xy[:, 1]
        cup_pos[:, 2] = deck_z + self._cup_bbox_half_z[env_idx] + 0.001
        self.cup.set_pose(Pose.create_from_pq(cup_pos, upright.clone()))

        ball_pos = cup_pos.clone()
        ball_pos[:, 2] = cup_pos[:, 2] + self._cup_floor_top_local_z + self._payload_radius
        ball_pos[:, :2] = ball_pos[:, :2] + self._sample_payload_offset(env_idx, b)
        self.ball.set_pose(Pose.create_from_pq(ball_pos, upright.clone()))

        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        zero = torch.zeros((b, 3), device=device)
        self.cup.set_linear_velocity(vel)
        self.cup.set_angular_velocity(zero)
        self.ball.set_linear_velocity(vel.clone())
        self.ball.set_angular_velocity(zero.clone())

        self._reset_decant_trackers(env_idx)

    def _sample_cup_center_xy(self, env_idx: torch.Tensor, b: int, upstream: torch.Tensor) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        device = self.device
        xy = torch.zeros((b, 2), device=device)
        xy[:, 0] = ox
        xy[:, 1] = oy - upstream * self.belt.direction_sign[env_idx]
        return xy

    def _sample_payload_offset(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        offset = torch.zeros((b, 2), device=device)
        rcfg = self.randomization
        if not rcfg.payload_offset:
            return offset
        free_x = self._cup_inner_half_x - self._payload_radius
        free_y = self._cup_inner_half_y[env_idx] - self._payload_radius
        u = torch.as_tensor(self._episode_rng.uniform(-1.0, 1.0, size=(b, 2)).astype(np.float32), device=device)
        offset[:, 0] = u[:, 0] * rcfg.payload_offset_frac * free_x
        offset[:, 1] = u[:, 1] * rcfg.payload_offset_frac * free_y
        return offset

    def _sample_bin_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        device = self.device
        rcfg = self.randomization
        pos = torch.zeros((b, 3), device=device)
        if rcfg.bin_side:
            xs = self._episode_rng.uniform(*rcfg.bin_x_range, size=b).astype(np.float32)
            y_mag = self._episode_rng.uniform(*rcfg.bin_side_y_range, size=b).astype(np.float32)
            side = self._episode_rng.choice([-1.0, 1.0], size=b).astype(np.float32)
            pos[:, 0] = ox + torch.as_tensor(xs, device=device)
            pos[:, 1] = oy + torch.as_tensor(side * y_mag, device=device)
        else:
            speed = self.belt.velocity[env_idx].abs()
            fast_band = (speed >= self._bin_fast_band_speed) | (speed == 0.0)
            grasp_rel = torch.where(fast_band, 1.0, -1.0)
            pos[:, 0] = ox + self._bin_offset_x
            pos[:, 1] = oy + self._bin_offset_y_mag * grasp_rel * self.belt.direction_sign[env_idx]
        pos[:, 2] = oz + self._bin_origin_above_surface
        return pos

    def _reset_decant_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_decant_sustain_counter"):
            self._decant_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._ball_touched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._ball_exited = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._shaken_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._prev_cup_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._prev_tcp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._decant_sustain_counter[env_idx] = 0
        self._ball_touched[env_idx] = False
        self._ball_exited[env_idx] = False
        self._shaken_out[env_idx] = False
        self._prev_cup_pos[env_idx] = self.cup.pose.p[env_idx]
        self._prev_tcp_pos[env_idx] = cast("Panda", self.agent).tcp.pose.p[env_idx]

    def evaluate(self):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        cup_pos = self.cup.pose.p
        ball_pos = self.ball.pose.p
        bin_pos = self.bin.pose.p
        rot = quaternion_to_matrix(self.cup.pose.q)

        f1 = torch.linalg.norm(self.scene.get_pairwise_contact_forces(agent.finger1_link, self.ball), dim=-1)
        f2 = torch.linalg.norm(self.scene.get_pairwise_contact_forces(agent.finger2_link, self.ball), dim=-1)
        self._ball_touched = self._ball_touched | (f1 > self._ball_touch_force_threshold)
        self._ball_touched = self._ball_touched | (f2 > self._ball_touch_force_threshold)

        local = torch.einsum("bji,bj->bi", rot, ball_pos - cup_pos)
        s = self._payload_box_slack
        in_cup = (
            (torch.abs(local[:, 0]) < self._cup_inner_half_x + s)
            & (torch.abs(local[:, 1]) < self._cup_inner_half_y + s)
            & (local[:, 2] > self._cup_floor_top_local_z - s)
            & (local[:, 2] < self._cup_half_z + s)
        )
        cup_up_cos = rot[:, 2, 2]
        poured_over = cup_up_cos < self._tip_exit_cos
        just_left = (~in_cup) & (~self._ball_exited)
        self._shaken_out = self._shaken_out | (just_left & ~poured_over)
        self._ball_exited = self._ball_exited | just_left

        rel = ball_pos - bin_pos
        in_bin_xy = (torch.abs(rel[:, 0]) < self._bin_inner_half_x) & (torch.abs(rel[:, 1]) < self._bin_inner_half_y)
        in_bin_z = (rel[:, 2] > self._bin_floor_half_height) & (rel[:, 2] < self._bin_rim_local_z)
        ball_settled = torch.linalg.norm(self.ball.linear_velocity, dim=-1) < self._payload_settled_speed

        finger_gap = self.gripper_aperture()
        width = self._cup_grasp_width
        cup_gripped = (finger_gap > width - self._grip_gap_squeeze_max) & (
            finger_gap < width + self._grip_gap_slack_max
        )
        advanced = self._eval_advanced("decant_pose")
        cup_rel_tcp = ((cup_pos - tcp_pos) - (self._prev_cup_pos - self._prev_tcp_pos)) * self.control_freq
        self._prev_cup_pos = torch.where(advanced.unsqueeze(-1), cup_pos, self._prev_cup_pos)
        self._prev_tcp_pos = torch.where(advanced.unsqueeze(-1), tcp_pos, self._prev_tcp_pos)
        cup_carried = torch.linalg.norm(cup_rel_tcp, dim=-1) < self._cup_carry_speed_match
        cup_in_hand = torch.linalg.norm(cup_pos - tcp_pos, dim=-1) < self._cup_hold_dist
        cup_above_rim = cup_pos[:, 2] >= bin_pos[:, 2] + self._bin_rim_local_z

        ball_out_of_cup = ~in_cup

        holding = (
            in_bin_xy
            & in_bin_z
            & ball_settled
            & cup_gripped
            & cup_carried
            & cup_in_hand
            & cup_above_rim
            & ball_out_of_cup
        )

        self._decant_sustain_counter = self._advance_sustain("decant_sustain", self._decant_sustain_counter, holding)
        success = (self._decant_sustain_counter >= self._sustain_steps) & (~self._ball_touched) & (~self._shaken_out)

        g1 = torch.linalg.norm(self.scene.get_pairwise_contact_forces(agent.finger1_link, self.cup), dim=-1)
        g2 = torch.linalg.norm(self.scene.get_pairwise_contact_forces(agent.finger2_link, self.cup), dim=-1)

        return {
            "success": success,
            "in_bin_xy": in_bin_xy,
            "in_bin_z": in_bin_z,
            "ball_settled": ball_settled,
            "cup_gripped": cup_gripped,
            "cup_carried": cup_carried,
            "cup_in_hand": cup_in_hand,
            "cup_above_rim": cup_above_rim,
            "ball_out_of_cup": ball_out_of_cup,
            "ball_touched": self._ball_touched,
            "shaken_out": self._shaken_out,
            "ball_exited": self._ball_exited,
            "decant_sustain": self._decant_sustain_counter,
            "cup_up_cos": cup_up_cos,
            "finger_gap": finger_gap,
            "cup_grip_force": torch.minimum(g1, g2),
            "ball_to_bin_xy": torch.linalg.norm(rel[:, :2], dim=-1),
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        cup_pos = self.cup.pose.p
        bin_pos = self.bin.pose.p

        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - cup_pos, dim=-1))
        carry_reward = 1 - torch.tanh(3 * torch.linalg.norm(cup_pos[:, :2] - bin_pos[:, :2], dim=-1))
        pour_reward = ((1.0 - info["cup_up_cos"]) * 0.5).clamp(0.0, 1.0)
        clean = (~info["ball_touched"] & ~info["shaken_out"]).float()

        in_bin = (info["in_bin_xy"] & info["in_bin_z"]).float()
        return reach_reward + carry_reward + clean * (pour_reward + 2.0 * in_bin)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
