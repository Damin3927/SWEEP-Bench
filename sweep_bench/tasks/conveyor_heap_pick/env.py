from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.materials import flat
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_heap_pick.assets import build_corral
from sweep_bench.tasks.conveyor_heap_pick.config import RandomizationConfig
from sweep_bench.tasks.conveyor_heap_pick.scripted import ScriptedHeapPickMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor

_SLOT_PITCH = 0.0502
_MEMBER_SLOTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (_SLOT_PITCH, 0.0),
    (-_SLOT_PITCH, 0.0),
    (0.0, _SLOT_PITCH),
    (0.0, -_SLOT_PITCH),
)

_RISER_COLOR = (0.72, 0.68, 0.60, 1.0)
_FLAT_COLORS: tuple[tuple[float, float, float, float], ...] = (
    (0.70, 0.66, 0.58, 1.0),
    (0.68, 0.68, 0.62, 1.0),
    (0.72, 0.66, 0.62, 1.0),
    (0.68, 0.70, 0.60, 1.0),
)


@bench_task("ConveyorHeapPick-v0")
@register_env("ConveyorHeapPick-v0", max_episode_steps=300)
class ConveyorHeapPickEnv(ScriptedHeapPickMixin, ConveyorWorkspaceEnv):
    """Pick a part out of a close-packed cluster on the belt without shifting the rest."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _conveyor_half_width: float = 0.12

    _corral_inner_half: float = 0.096
    _corral_wall_half_thickness: float = 0.005
    _corral_wall_half_height: float = 0.006
    _corral_floor_half_height: float = 0.004
    _corral_pad_half_height: float = 0.002
    _corral_pad_static_friction: float = 0.9
    _corral_pad_dynamic_friction: float = 0.8
    _corral_patch_radius: float = 0.08

    _member_half_xy: float = 0.025
    _riser_half_z: float = 0.045
    _flat_half_z: float = 0.007
    _riser_density: float = 2000.0
    _flat_density: float = 7800.0

    _grasp_width_tol: float = 0.008
    _carry_speed_match_threshold: float = 0.03
    _lift_clearance: float = 0.10
    _pick_sustain_steps: int = 6

    _disturb_tol: float = 0.015
    _carrier_travel_frac: float = 0.5
    _disturb_lift_tol: float = 0.006

    _carrier_stall_limit: int = 10
    _carrier_z_tol: float = 0.006
    _carrier_tilt_deg: float = 10.0
    _static_carrier_travel_tol: float = 0.015
    _carrier_track_tol: float = 0.08

    default_prompt: str = "lift a part out of the packed tray on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _corral_outer_half(self) -> float:
        return self._corral_inner_half + 2.0 * self._corral_wall_half_thickness

    @property
    def _corral_floor_local_z(self) -> float:
        return self._corral_floor_half_height

    @property
    def _corral_rim_local_z(self) -> float:
        return self._corral_floor_half_height + 2.0 * self._corral_wall_half_height

    @property
    def _member_clear_local_z(self) -> float:
        return max(self._corral_rim_local_z, self._corral_floor_local_z + 2.0 * self._flat_half_z)

    @property
    def _corral_rest_z(self) -> float:
        return self.workspace_offset[2] + self._conveyor_deck_height + self._corral_floor_half_height

    @property
    def _member_grasp_width(self) -> float:
        return 2.0 * self._member_half_xy

    def _corral_local_world_z(self, local_z: float) -> torch.Tensor:
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 2] = local_z
        return (self.corral.pose * Pose.create_from_pq(local)).p[:, 2]

    def corral_clear_world_z(self) -> torch.Tensor:
        return self._corral_local_world_z(self._member_clear_local_z)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        corral_builder = build_corral(
            self.scene,
            inner_half_x=self._corral_inner_half,
            inner_half_y=self._corral_inner_half,
            wall_half_thickness=self._corral_wall_half_thickness,
            wall_half_height=self._corral_wall_half_height,
            floor_half_height=self._corral_floor_half_height,
            pad_half_height=self._corral_pad_half_height,
            pad_static_friction=self._corral_pad_static_friction,
            pad_dynamic_friction=self._corral_pad_dynamic_friction,
        )
        corral_builder.initial_pose = sapien.Pose(p=[ox, oy, self._corral_rest_z])
        self.corral = corral_builder.build(name="heap_corral")
        self.belt.register_rider(self.corral, patch_radius=self._corral_patch_radius)

        self.riser = self._build_member(
            "heap_riser", self._riser_half_z, self._riser_density, _RISER_COLOR, deck_z + 0.40
        )
        self.flats: list[Actor] = [
            self._build_member(
                f"heap_flat_{i}", self._flat_half_z, self._flat_density, color, deck_z + 0.40 + 0.05 * (i + 1)
            )
            for i, color in enumerate(_FLAT_COLORS)
        ]
        self.members: list[Actor] = [self.riser, *self.flats]

        self._member_half_z = torch.as_tensor(
            [self._riser_half_z] + [self._flat_half_z] * len(self.flats),
            dtype=torch.float32,
            device=self.device,
        )
        self._slot_xy = torch.as_tensor(_MEMBER_SLOTS, dtype=torch.float32, device=self.device)

    def _build_member(
        self, name: str, half_z: float, density: float, color: tuple[float, float, float, float], park_z: float
    ):
        ox, oy, _oz = self.workspace_offset
        half = [self._member_half_xy, self._member_half_xy, half_z]
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=half, density=density)
        builder.add_box_visual(half_size=half, material=flat(color))
        builder.initial_pose = sapien.Pose(p=[ox, oy, park_z])
        actor = builder.build(name=name)
        actor.angular_damping = 5.0
        return actor

    @property
    def _primary_rider(self) -> Actor:
        return self.riser

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        corral_pos = self._sample_corral_position(env_idx, b)
        corral_q = torch.zeros((b, 4), device=device)
        corral_q[:, 0] = 1.0
        corral_pose = Pose.create_from_pq(corral_pos, corral_q)
        self.corral.set_pose(corral_pose)
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        zero = torch.zeros((b, 3), device=device)
        self.corral.set_linear_velocity(vel)
        self.corral.set_angular_velocity(zero)

        slot_of_member = self._assign_slots(b)
        seats = torch.zeros((b, len(self.members), 3), device=device)
        member_q = torch.zeros((b, 4), device=device)
        member_q[:, 0] = 1.0
        for j, member in enumerate(self.members):
            local = torch.zeros((b, 3), device=device)
            local[:, :2] = self._slot_xy[slot_of_member[:, j]]
            local[:, 2] = self._corral_floor_local_z + self._member_half_z[j] + 0.0005
            seats[:, j] = local
            member.set_pose(Pose.create_from_pq((corral_pose * Pose.create_from_pq(local)).p, member_q))
            member.set_linear_velocity(vel)
            member.set_angular_velocity(zero)

        self._reset_heap_trackers(env_idx, seats)

    def _assign_slots(self, b: int) -> torch.Tensor:
        device = self.device
        n = len(self.members)
        if self.randomization.riser_slot:
            draws = self._episode_rng.randint(0, n, size=b)
            riser_slot = torch.as_tensor(draws.astype(np.int64), device=device)
        else:
            riser_slot = torch.zeros(b, dtype=torch.long, device=device)
        slots = torch.arange(n, device=device).expand(b, n)
        rest = slots[slots != riser_slot.unsqueeze(-1)].view(b, n - 1)
        return torch.cat([riser_slot.unsqueeze(-1), rest], dim=-1)

    def _sample_corral_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        device = self.device
        start_y = self._conveyor_half_length - self._corral_outer_half - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.corral_lateral:
            lo, hi = self.randomization.lateral_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = pos[:, 0] + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = self._corral_rest_z + 0.001
        return pos

    def _reset_heap_trackers(self, env_idx: torch.Tensor, seats: torch.Tensor) -> None:
        device = self.device
        n = len(self.members)
        if not hasattr(self, "_pick_sustain"):
            self._pick_sustain = torch.zeros(self.num_envs, dtype=torch.long, device=device)
            self._member_seat = torch.zeros((self.num_envs, n, 3), device=device)
            self._member_moved = torch.zeros((self.num_envs, n), dtype=torch.bool, device=device)
            self._member_shoved_in_cluster = torch.zeros((self.num_envs, n), dtype=torch.bool, device=device)
            self._member_was_in_cluster = torch.ones((self.num_envs, n), dtype=torch.bool, device=device)
            self._corral_spawn_xy = torch.zeros((self.num_envs, 2), device=device)
            self._corral_stall_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
            self._corral_stalled = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self._pre_grasp_riding = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self._was_grasping = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self._caught_moving = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self._member_disp_peak = torch.zeros((self.num_envs, n), device=device)
            self._member_lift_peak = torch.zeros((self.num_envs, n), device=device)
            self._pick_done = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self._picked_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)
            self._prev_member_pos = torch.zeros((self.num_envs, n, 3), device=device)
            self._prev_tcp_pos = torch.zeros((self.num_envs, 3), device=device)
        self._pick_sustain[env_idx] = 0
        self._member_seat[env_idx] = seats
        self._member_moved[env_idx] = False
        self._member_shoved_in_cluster[env_idx] = False
        self._member_was_in_cluster[env_idx] = True
        self._corral_stall_steps[env_idx] = 0
        self._corral_stalled[env_idx] = False
        self._pre_grasp_riding[env_idx] = True
        self._was_grasping[env_idx] = False
        self._caught_moving[env_idx] = True
        self._corral_spawn_xy[env_idx] = self.corral.pose.p[env_idx, :2]
        self._member_disp_peak[env_idx] = 0.0
        self._member_lift_peak[env_idx] = 0.0
        self._pick_done[env_idx] = False
        self._picked_idx[env_idx] = -1
        self._prev_member_pos[env_idx] = self._member_world_pos()[env_idx]
        self._prev_tcp_pos[env_idx] = cast("Panda", self.agent).tcp.pose.p[env_idx]
        self._heap_ready = True

    def _corral_half_extent(self) -> tuple[torch.Tensor, torch.Tensor]:
        rot = quaternion_to_matrix(self.corral.pose.q)
        half = torch.tensor([self._corral_outer_half, self._corral_outer_half], device=self.device)
        span = rot[:, :2, :2].abs() @ half
        return span[:, 0], span[:, 1]

    def _corral_support_centre(self) -> torch.Tensor:
        return quaternion_to_matrix(self.corral.pose.q)[:, :2, 2] * (-self._corral_floor_half_height)

    def _member_half_extent_in_corral(self) -> torch.Tensor:
        corral_rot = quaternion_to_matrix(self.corral.pose.q)
        rows = torch.stack(
            [
                torch.bmm(quaternion_to_matrix(member.pose.q).transpose(1, 2), corral_rot[:, :, 2].unsqueeze(-1))
                .squeeze(-1)
                .abs()
                for member in self.members
            ],
            dim=1,
        )
        half = torch.stack(
            [
                torch.full_like(self._member_half_z, self._member_half_xy),
                torch.full_like(self._member_half_z, self._member_half_xy),
                self._member_half_z,
            ],
            dim=-1,
        )
        return (rows * half).sum(dim=-1)

    def _member_world_pos(self) -> torch.Tensor:
        return torch.stack([m.pose.p for m in self.members], dim=1)

    def _member_corral_pos(self) -> torch.Tensor:
        inv = self.corral.pose.inv()
        return torch.stack([(inv * m.pose).p for m in self.members], dim=1)

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._latch_disturbance()

    def _latch_disturbance(self) -> None:
        if not getattr(self, "_heap_ready", False):
            return
        delta = self._member_corral_pos() - self._member_seat
        disp_h = torch.linalg.norm(delta[:, :, :2], dim=-1)
        disp_v = delta[:, :, 2].abs()
        watching = (~self._pick_done).unsqueeze(1)
        self._member_disp_peak = torch.where(
            watching, torch.maximum(self._member_disp_peak, disp_h), self._member_disp_peak
        )
        self._member_lift_peak = torch.where(
            watching, torch.maximum(self._member_lift_peak, disp_v), self._member_lift_peak
        )
        self._member_moved = self._member_moved | (
            watching & ((disp_h > self._disturb_tol) | (disp_v > self._disturb_lift_tol))
        )
        local = self._member_corral_pos()
        in_cluster = (local[:, :, 2] - self._member_half_extent_in_corral()) <= self._member_clear_local_z
        in_window = in_cluster | self._member_was_in_cluster
        self._member_was_in_cluster = in_cluster
        agent = cast("Panda", self.agent)
        grasped = torch.stack([agent.is_grasping(m) for m in self.members], dim=1)
        self._member_shoved_in_cluster = self._member_shoved_in_cluster | (
            watching & in_window & ((disp_h > self._disturb_tol) | (~grasped & (disp_v > self._disturb_lift_tol)))
        )

    def evaluate(self):
        ox, oy, _oz = self.workspace_offset
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        corral_pose = self.corral.pose
        corral_pos = corral_pose.p
        belt_v = self.belt.velocity
        member_pos = self._member_world_pos()

        width_matches = torch.abs(self.gripper_aperture() - self._member_grasp_width) <= self._grasp_width_tol
        advanced = self._eval_advanced("heap_carry")
        rel_vel = (
            (member_pos - tcp_pos.unsqueeze(1)) - (self._prev_member_pos - self._prev_tcp_pos.unsqueeze(1))
        ) * self.control_freq
        carried = torch.linalg.norm(rel_vel, dim=-1) < self._carry_speed_match_threshold
        self._prev_member_pos = torch.where(advanced[:, None, None], member_pos, self._prev_member_pos)
        self._prev_tcp_pos = torch.where(advanced[:, None], tcp_pos, self._prev_tcp_pos)
        grasped = torch.stack([agent.is_grasping(m) for m in self.members], dim=1)
        member_held = width_matches.unsqueeze(1) & grasped & carried
        member_lifted = member_pos[:, :, 2] >= self._corral_rest_z + self._corral_rim_local_z + self._lift_clearance
        member_clear = member_held & member_lifted
        picked_now = member_clear.any(dim=1)

        self._latch_disturbance()
        first_clear = member_clear.float().argmax(dim=1)
        self._picked_idx = torch.where(advanced & picked_now & (self._picked_idx < 0), first_clear, self._picked_idx)
        exempt = torch.nn.functional.one_hot(self._picked_idx.clamp(min=0), len(self.members)).bool()
        exempt = exempt & (self._picked_idx >= 0).unsqueeze(-1)
        others_clean = ~((self._member_moved & ~exempt) | (self._member_shoved_in_cluster & exempt)).any(dim=1)

        carrier_z_ok = torch.abs(corral_pos[:, 2] - self._corral_rest_z) <= self._carrier_z_tol
        half_x, half_y = self._corral_half_extent()
        support = corral_pos[:, :2] + self._corral_support_centre()
        carrier_x_ok = torch.abs(support[:, 0] - ox) <= self._conveyor_half_width - half_x
        carrier_upright = quaternion_to_matrix(corral_pose.q)[:, 2, 2] >= math.cos(math.radians(self._carrier_tilt_deg))
        carrier_on_belt = torch.abs(support[:, 1] - oy) <= self._conveyor_half_length - half_y
        corral_v_y = self.corral.linear_velocity[:, 1]
        carrier_tracking = torch.abs(corral_v_y - belt_v) <= self._carrier_track_tol
        carrier_travelling = self.belt_carried(self.corral, self._carrier_travel_frac)
        corral_travel = torch.linalg.norm(corral_pos[:, :2] - self._corral_spawn_xy, dim=-1)
        carrier_stationary = (torch.abs(belt_v) > 0.0) | (corral_travel <= self._static_carrier_travel_tol)
        carrier_ok = (
            carrier_z_ok
            & carrier_x_ok
            & carrier_upright
            & carrier_on_belt
            & carrier_tracking
            & carrier_travelling
            & carrier_stationary
        )

        grasping_any = member_held.any(dim=1)
        pre_grasp = advanced & ~grasping_any
        self._pre_grasp_riding = torch.where(pre_grasp, carrier_travelling & carrier_z_ok, self._pre_grasp_riding)
        first_grasp = grasping_any & ~self._was_grasping
        new_attempt = first_grasp & ~(member_lifted.any(dim=1))
        regrasp = first_grasp & (member_lifted.any(dim=1))
        self._caught_moving = torch.where(new_attempt, self._pre_grasp_riding, self._caught_moving)
        self._caught_moving = torch.where(regrasp, self._caught_moving & self._pre_grasp_riding, self._caught_moving)
        self._was_grasping = torch.where(advanced, grasping_any, self._was_grasping)

        capturing = ~picked_now
        self._corral_stall_steps = self._corral_stall_steps + (capturing & ~carrier_travelling).long() * advanced.long()
        self._corral_stalled = self._corral_stalled | (self._corral_stall_steps >= self._carrier_stall_limit)
        carrier_never_stalled = ~self._corral_stalled

        self._pick_sustain = self._advance_sustain("heap_pick_sustain", self._pick_sustain, picked_now & carrier_ok)
        sustained = self._pick_sustain >= self._pick_sustain_steps
        self._pick_done = self._pick_done | sustained
        success = sustained & others_clean & carrier_never_stalled & self._caught_moving

        return {
            "success": success,
            "picked": picked_now,
            "picked_idx": self._picked_idx,
            "riser_held": member_held[:, 0],
            "riser_lifted": member_lifted[:, 0],
            "others_clean": others_clean,
            "member_moved": self._member_moved,
            "member_disp_peak": self._member_disp_peak.amax(dim=1),
            "others_disp_peak": torch.where(
                exempt, torch.zeros_like(self._member_disp_peak), self._member_disp_peak
            ).amax(dim=1),
            "others_lift_peak": torch.where(
                exempt, torch.zeros_like(self._member_lift_peak), self._member_lift_peak
            ).amax(dim=1),
            "carrier_ok": carrier_ok,
            "carrier_z_ok": carrier_z_ok,
            "carrier_x_ok": carrier_x_ok,
            "carrier_upright": carrier_upright,
            "carrier_on_belt": carrier_on_belt,
            "carrier_tracking": carrier_tracking,
            "carrier_travelling": carrier_travelling,
            "carrier_stationary": carrier_stationary,
            "carrier_never_stalled": carrier_never_stalled,
            "caught_moving": self._caught_moving,
            "pick_sustain": self._pick_sustain,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        riser_pos = self.riser.pose.p

        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - riser_pos, dim=-1))
        above_rim = riser_pos[:, 2] - (self._corral_rest_z + self._corral_rim_local_z)
        lift_reward = torch.clamp(above_rim / self._lift_clearance, 0, 1)
        earned = (
            info["riser_held"] & info["others_clean"] & info["carrier_never_stalled"] & info["caught_moving"]
        ).float()
        return reach_reward + 2.0 * lift_reward * earned

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
