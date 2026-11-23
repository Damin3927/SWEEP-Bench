from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_cap_swipe.assets import build_knob_cap, build_seat_block
from sweep_bench.tasks.conveyor_cap_swipe.config import RandomizationConfig
from sweep_bench.tasks.conveyor_cap_swipe.scripted import ScriptedCapSwipeMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorCapSwipe-v0")
@register_env("ConveyorCapSwipe-v0", max_episode_steps=300)
class ConveyorCapSwipeEnv(ScriptedCapSwipeMixin, ConveyorWorkspaceEnv):
    """Swipe a loose cover off a passing carrier without disturbing the carrier."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _carrier_half_x: float = 0.040
    _carrier_half_y: float = 0.050
    _carrier_seat_half: float = 0.029
    _carrier_seat_depth: float = 0.002
    _carrier_floor_half_height: float = 0.020
    _carrier_density: float = 21000.0

    _carrier_patch_radius: float = 0.035

    _knob_radius: float = 0.014
    _knob_half_height: float = 0.024
    _flange_radius: float = 0.028
    _flange_half_thickness: float = 0.004
    _cover_density: float = 500.0
    _cover_friction: float = 0.3
    _cover_angular_damping: float = 8.0

    _cover_drop_below_deck: float = 0.10
    _cover_settle_speed: float = 0.05
    _carrier_max_tilt_deg: float = 15.0
    _carrier_lift_tol: float = 0.008
    _carrier_shove_tol: float = 0.030
    _cover_unseat_radius: float = 0.020
    _cover_contact_radius: float = 0.050
    _cap_swipe_sustain_steps: int = 4

    default_prompt: str = "swipe the cap off the block on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _carrier_height(self) -> float:
        return 2.0 * self._carrier_floor_half_height + self._carrier_seat_depth

    @property
    def _carrier_origin_above_deck(self) -> float:
        return self._carrier_floor_half_height

    @property
    def _cover_half_height(self) -> float:
        return self._flange_half_thickness + self._knob_half_height

    @property
    def _cover_rest_origin_above_deck(self) -> float:
        return 2.0 * self._carrier_floor_half_height + self._cover_half_height

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        carrier_builder = build_seat_block(
            self.scene,
            half_x=self._carrier_half_x,
            half_y=self._carrier_half_y,
            floor_half_height=self._carrier_floor_half_height,
            seat_half=self._carrier_seat_half,
            seat_depth=self._carrier_seat_depth,
            density=self._carrier_density,
        )
        carrier_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._carrier_origin_above_deck])
        self.carrier = carrier_builder.build(name="cap_swipe_carrier")
        self.belt.register_rider(self.carrier, patch_radius=self._carrier_patch_radius)

        cover_builder = build_knob_cap(
            self.scene,
            knob_radius=self._knob_radius,
            knob_half_height=self._knob_half_height,
            flange_radius=self._flange_radius,
            flange_half_thickness=self._flange_half_thickness,
            density=self._cover_density,
            friction=self._cover_friction,
        )
        cover_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.cover = cover_builder.build(name="cap_swipe_cover")
        for body in self.cover._bodies:
            body.angular_damping = self._cover_angular_damping

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        carrier_pos = self._sample_carrier_position(env_idx, b)
        upright_q = torch.zeros((b, 4), device=device)
        upright_q[:, 0] = 1.0
        self.carrier.set_pose(Pose.create_from_pq(carrier_pos, upright_q))
        carrier_vel = torch.zeros((b, 3), device=device)
        carrier_vel[:, 1] = self.belt.velocity[env_idx]
        self.carrier.set_linear_velocity(carrier_vel)
        self.carrier.set_angular_velocity(torch.zeros((b, 3), device=device))

        cover_pos = carrier_pos.clone()
        cover_pos[:, :2] = cover_pos[:, :2] + self._sample_cover_offset(b)
        cover_pos[:, 2] = deck_z + self._cover_rest_origin_above_deck
        self.cover.set_pose(Pose.create_from_pq(cover_pos, upright_q.clone()))
        self.cover.set_linear_velocity(carrier_vel.clone())
        self.cover.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._sample_swipe_side(env_idx, b)
        self._reset_cap_swipe_trackers(env_idx, carrier_pos[:, 0])

    def _sample_carrier_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._carrier_half_y - 0.015
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.carrier_lateral_offset:
            lo, hi = self.randomization.lateral_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = pos[:, 0] + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._carrier_origin_above_deck
        return pos

    def _sample_cover_offset(self, b: int) -> torch.Tensor:
        device = self.device
        if not self.randomization.cover_offset:
            return torch.zeros((b, 2), device=device)
        play = self._carrier_seat_half - self._flange_radius
        off = self._episode_rng.uniform(-play, play, size=(b, 2)).astype(np.float32)
        return torch.as_tensor(off, device=device)

    def _sample_swipe_side(self, env_idx: torch.Tensor, b: int) -> None:
        if not hasattr(self, "_cap_swipe_push_sign"):
            self._cap_swipe_push_sign = torch.ones((self.num_envs,), device=self.device, dtype=torch.float32)
        if self.randomization.swipe_side:
            signs = self._episode_rng.choice(np.array([1.0, -1.0], dtype=np.float32), size=b)
            self._cap_swipe_push_sign[env_idx] = torch.as_tensor(signs, device=self.device)
        else:
            self._cap_swipe_push_sign[env_idx] = 1.0

    def _reset_cap_swipe_trackers(self, env_idx: torch.Tensor, spawn_x: torch.Tensor | None = None) -> None:
        if not hasattr(self, "_cap_swipe_sustain_counter"):
            zeros = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            falses = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._cap_swipe_sustain_counter = zeros
            self._cover_unseated_seen = falses.clone()
            self._cover_contact_caused = falses.clone()
            self._cover_tcp_was_near = falses.clone()
            self._carrier_spawn_x = torch.zeros(self.num_envs, device=self.device)
        self._cap_swipe_sustain_counter[env_idx] = 0
        self._cover_unseated_seen[env_idx] = False
        self._cover_contact_caused[env_idx] = False
        self._cover_tcp_was_near[env_idx] = False
        if spawn_x is not None:
            self._carrier_spawn_x[env_idx] = spawn_x

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        cover_pos = self.cover.pose.p
        carrier_pos = self.carrier.pose.p

        cover_down = cover_pos[:, 2] < deck_z - self._cover_drop_below_deck
        cover_settled = torch.linalg.norm(self.cover.linear_velocity, dim=-1) < self._cover_settle_speed

        upright_cos = quaternion_to_matrix(self.carrier.pose.q)[:, 2, 2]
        carrier_upright = upright_cos > math.cos(math.radians(self._carrier_max_tilt_deg))
        carrier_riding = (
            torch.abs(carrier_pos[:, 2] - (deck_z + self._carrier_origin_above_deck)) < self._carrier_lift_tol
        )
        carrier_on_belt = torch.abs(carrier_pos[:, 1] - oy) < self._conveyor_half_length
        carrier_undisplaced = torch.abs(carrier_pos[:, 0] - self._carrier_spawn_x) < self._carrier_shove_tol

        unseated = torch.linalg.norm(cover_pos[:, :2] - carrier_pos[:, :2], dim=-1) > self._cover_unseat_radius
        tcp_near = torch.linalg.norm(tcp_pos - cover_pos, dim=-1) < self._cover_contact_radius
        first_unseat = unseated & ~self._cover_unseated_seen
        self._cover_contact_caused = torch.where(first_unseat, self._cover_tcp_was_near, self._cover_contact_caused)
        self._cover_unseated_seen = self._cover_unseated_seen | unseated
        self._cover_tcp_was_near = tcp_near

        holding = cover_down & cover_settled & carrier_upright & carrier_riding & carrier_on_belt & carrier_undisplaced
        self._cap_swipe_sustain_counter = self._advance_sustain(
            "cap_swipe_sustain", self._cap_swipe_sustain_counter, holding
        )
        sustained = self._cap_swipe_sustain_counter >= self._cap_swipe_sustain_steps

        success = sustained & self._cover_contact_caused

        return {
            "success": success,
            "cover_down": cover_down,
            "cover_settled": cover_settled,
            "cover_unseated": self._cover_unseated_seen,
            "contact_caused": self._cover_contact_caused,
            "carrier_upright": carrier_upright,
            "carrier_riding": carrier_riding,
            "carrier_on_belt": carrier_on_belt,
            "carrier_undisplaced": carrier_undisplaced,
            "carrier_band_margin": self._conveyor_half_width - self._carrier_half_x - torch.abs(carrier_pos[:, 0] - ox),
            "cap_swipe_sustain": self._cap_swipe_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        cover_pos = self.cover.pose.p
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        tcp_to_cover = torch.linalg.norm(tcp_pos - cover_pos, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_cover)

        seat_z = deck_z + self._cover_rest_origin_above_deck
        span = seat_z - (deck_z - self._cover_drop_below_deck)
        drop_reward = ((seat_z - cover_pos[:, 2]) / span).clamp(0.0, 1.0)

        undisturbed = (info["carrier_upright"] & info["carrier_riding"] & info["carrier_undisplaced"]).float()
        return reach_reward + 2.0 * drop_reward * info["contact_caused"].float() * undisturbed

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
