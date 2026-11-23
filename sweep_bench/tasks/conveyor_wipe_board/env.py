from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply, quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import carried_speed
from sweep_bench.tasks.conveyor_wipe_board.assets import (
    build_mark_decal,
    build_marker_board,
    build_wipe_pad,
    mark_wear_colors,
)
from sweep_bench.tasks.conveyor_wipe_board.config import RandomizationConfig
from sweep_bench.tasks.conveyor_wipe_board.scripted import ScriptedWipeBoardMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor

_UNBOUNDED = 1.0e6


def _slab(offset: torch.Tensor, rate: torch.Tensor, half: float) -> tuple[torch.Tensor, torch.Tensor]:
    moving = rate.abs() > 1e-9
    denom = torch.where(moving, rate, torch.ones_like(rate))
    t0 = (offset - half) / denom
    t1 = (offset + half) / denom
    unbounded = torch.full_like(t0, _UNBOUNDED)
    inside = offset.abs() <= half
    lo = torch.where(moving, torch.minimum(t0, t1), torch.where(inside, -unbounded, unbounded))
    hi = torch.where(moving, torch.maximum(t0, t1), torch.where(inside, unbounded, -unbounded))
    return lo, hi


_GRIP_TO_PAD_Q = (0.5, -0.5, -0.5, -0.5)


@bench_task("ConveyorWipeBoard-v0")
@register_env("ConveyorWipeBoard-v0", max_episode_steps=300)
class ConveyorWipeBoardEnv(ScriptedWipeBoardMixin, ConveyorWorkspaceEnv):
    """Wipe the marks off a board riding the belt with a HELD pad, then lift off."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _board_half_x: float = 0.060
    _board_half_y: float = 0.075
    _board_half_thickness: float = 0.006
    _board_density: float = 2500.0
    _board_patch_radius: float = 0.032

    _mark_radius: float = 0.009
    _mark_offsets: tuple[tuple[float, float], ...] = ((0.012, 0.0), (-0.012, 0.0))
    _clear_path: float = 0.045
    _clear_count_required: int = 2
    _max_step_path: float = 0.05
    _mark_wear_tiers: int = 3
    _mark_proud: float = 0.0005

    _stem_half_grip: float = 0.020
    _stem_half_long: float = 0.014
    _stem_above_grip: float = 0.015
    _stem_below_grip: float = 0.045
    _face_half_across: float = 0.028
    _face_half_along: float = 0.011
    _face_half_thickness: float = 0.005
    _pad_density: float = 350.0

    _f_lo: float = 1.5
    _f_hi: float = 18.0
    _f_max_abs: float = 40.0
    _over_force_steps: int = 2

    _face_square_cos: float = math.cos(math.radians(25.0))
    _face_plane_tol: float = 0.006

    _pad_held_distance: float = 0.03
    _liftoff_tcp_clearance: float = 0.11
    _board_upright_cos: float = math.cos(math.radians(12.0))
    _board_travel_frac: float = 0.5
    _board_scrub_travel_frac: float = 0.25
    _wipe_sustain_steps: int = 8

    _board_spawn_max_dist: float = 0.80

    default_prompt: str = "wipe the marks off the passing board"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _board_top_above_deck(self) -> float:
        return 2.0 * self._board_half_thickness

    @property
    def _pad_face_below_tcp(self) -> float:
        return self._stem_below_grip + 2.0 * self._face_half_thickness

    @property
    def _pad_face_centre_below_grip(self) -> float:
        return self._stem_below_grip + self._face_half_thickness

    @property
    def _mark_count(self) -> int:
        return len(self._mark_offsets)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        board_builder = build_marker_board(
            self.scene,
            half_x=self._board_half_x,
            half_y=self._board_half_y,
            half_thickness=self._board_half_thickness,
            mark_radius=self._mark_radius,
            mark_offsets=self._mark_offsets,
            density=self._board_density,
        )
        board_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._board_top_above_deck])
        self.board = board_builder.build(name="marker_board")
        self.belt.register_rider(self.board, patch_radius=self._board_patch_radius)

        self.mark_decals: list[list[Actor]] = []
        for i, (mx, my) in enumerate(self._mark_offsets):
            tiers: list[Actor] = []
            for tier, color in enumerate(mark_wear_colors(self._mark_wear_tiers)):
                decal_builder = build_mark_decal(self.scene, radius=self._mark_radius, color=color)
                decal_builder.initial_pose = sapien.Pose(p=[ox + mx, oy + my, deck_z + self._board_top_above_deck])
                tiers.append(decal_builder.build_kinematic(name=f"mark_{i}_wear_{tier}"))
            self.mark_decals.append(tiers)

        pad_builder = build_wipe_pad(
            self.scene,
            stem_half_grip=self._stem_half_grip,
            stem_half_long=self._stem_half_long,
            stem_above_grip=self._stem_above_grip,
            stem_below_grip=self._stem_below_grip,
            face_half_across=self._face_half_across,
            face_half_along=self._face_half_along,
            face_half_thickness=self._face_half_thickness,
            density=self._pad_density,
        )
        pad_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.pad = pad_builder.build(name="wipe_pad")

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.pad]

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = int(env_idx.shape[0])
        return 2.0 * np.full((b, 1), self._stem_half_grip - 0.0005, dtype=np.float32)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        board_q = self._sample_board_orientation(env_idx, b)
        board_pos = self._sample_board_position(env_idx, b)
        self.board.set_pose(Pose.create_from_pq(board_pos, board_q))
        board_vel = torch.zeros((b, 3), device=device)
        board_vel[:, 1] = self.belt.velocity[env_idx]
        self.board.set_linear_velocity(board_vel)
        self.board.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pose = cast("Panda", self.agent).tcp.pose
        grip_to_pad = torch.tensor(_GRIP_TO_PAD_Q, device=device, dtype=torch.float32).expand(b, 4)
        pad_q = quaternion_multiply(tcp_pose.q[env_idx], grip_to_pad)
        self.pad.set_pose(Pose.create_from_pq(tcp_pose.p[env_idx], pad_q))
        self.pad.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.pad.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_wipe_trackers(env_idx, b)
        self._wipe_ready = True

    def _sample_board_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.board_yaw:
            lo, hi = self.randomization.board_yaw_range
            yaw = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            yaw_t = torch.as_tensor(yaw, device=device)
            q[:, 0] = torch.cos(yaw_t * 0.5)
            q[:, 3] = torch.sin(yaw_t * 0.5)
        return q

    def _sample_clear_path(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        if not self.randomization.clear_path:
            return torch.full((b,), self._clear_path, device=self.device)
        lo, hi = self.randomization.clear_path_range
        draw = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        return torch.as_tensor(draw, device=self.device) * self._clear_path

    def _sample_board_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = min(
            self._conveyor_half_length - self._board_half_y - 0.04,
            self._board_spawn_max_dist,
        )
        board_pos = torch.zeros((b, 3), device=device)
        board_pos[:, 0] = ox
        board_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        board_pos[:, 2] = deck_z + self._board_top_above_deck
        return board_pos

    def _update_abrasion(self) -> None:
        if not getattr(self, "_wipe_ready", False):
            return
        advanced = self._eval_advanced("wipe_abrasion")

        face = self._pad_face_in_board()
        local, height = face[:, :2], face[:, 2]
        prev, prev_height = self._pad_prev_local, self._pad_prev_height
        adv2 = advanced.unsqueeze(-1)
        self._pad_prev_local = torch.where(adv2, local, prev)
        self._pad_prev_height = torch.where(advanced, height, prev_height)
        step_path = torch.linalg.norm(local - prev, dim=-1)
        step_path = torch.where(step_path <= self._max_step_path, step_path, torch.zeros_like(step_path))

        world = self._pad_face_world()
        board_rot = quaternion_to_matrix(self.board.pose.q)
        arm_world = (world - self._pad_prev_world).unsqueeze(-1)
        self._pad_prev_world = torch.where(advanced.unsqueeze(-1), world, self._pad_prev_world)
        arm_path = torch.linalg.norm((board_rot.transpose(-1, -2) @ arm_world).squeeze(-1)[:, :2], dim=-1)
        passive_frac = (step_path - torch.minimum(step_path, arm_path)) / step_path.clamp(min=1e-9)

        press, _tangential = self._pad_contact()
        over = press > self._f_max_abs
        stepped = torch.where(over, self._over_force_run + 1, torch.zeros_like(self._over_force_run))
        self._over_force_run = torch.where(advanced, stepped, self._over_force_run)
        self._over_forced = self._over_forced | (self._over_force_run >= self._over_force_steps)
        working = (
            (press >= self._f_lo)
            & (press <= self._f_hi)
            & self._pad_in_hand()
            & self._face_on_board()
            & self._board_travelling(self._board_scrub_travel_frac)
            & self._board_upright()
        )

        covered = self._covered_fraction(prev, local, prev_height, height)
        credit = (step_path * working * advanced).unsqueeze(0) * covered

        passive = credit * passive_frac.unsqueeze(0)
        allowance = (2.0 * self._face_half_along - self._passive_credit).clamp(min=0.0)
        paid = torch.minimum(passive, allowance)
        self._passive_credit = self._passive_credit + paid
        scrubbed = credit - passive + paid
        self._dirt = (self._dirt - scrubbed).clamp(min=0.0)

    def _covered_fraction(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        start_height: torch.Tensor,
        end_height: torch.Tensor,
    ) -> torch.Tensor:
        along, across = self._face_axes_in_board()
        step = end - start
        rel = self._mark_local - start.unsqueeze(0)
        lo = torch.full(rel.shape[:2], -_UNBOUNDED, device=self.device)
        hi = torch.full(rel.shape[:2], _UNBOUNDED, device=self.device)
        for axis, half in ((along, self._face_half_along), (across, self._face_half_across)):
            offset = (rel * axis.unsqueeze(0)).sum(dim=-1)
            rate = (step * axis).sum(dim=-1).unsqueeze(0)
            axis_lo, axis_hi = _slab(offset, rate, half)
            lo = torch.maximum(lo, axis_lo)
            hi = torch.minimum(hi, axis_hi)
        offset = start_height - self._face_half_thickness
        rate = start_height - end_height
        height_lo, height_hi = _slab(offset.unsqueeze(0), rate.unsqueeze(0), self._face_plane_tol)
        lo = torch.maximum(lo, height_lo)
        hi = torch.minimum(hi, height_hi)
        return (hi.clamp(max=1.0) - lo.clamp(min=0.0)).clamp(min=0.0)

    def _board_upright(self) -> torch.Tensor:
        return quaternion_to_matrix(self.board.pose.q)[:, 2, 2] >= self._board_upright_cos

    def _board_travelling(self, frac: float, *, success_check: bool = False) -> torch.Tensor:
        belt_speed = torch.abs(self.belt.velocity)
        along_v = self.board.linear_velocity[:, 1] * self.belt.direction_sign
        min_along = carried_speed(belt_speed, frac) if success_check else frac * belt_speed
        return (along_v >= min_along) | (belt_speed <= 1e-6)

    def _face_axes_in_board(self) -> tuple[torch.Tensor, torch.Tensor]:
        pad_rot = quaternion_to_matrix(self.pad.pose.q)
        board_rot = quaternion_to_matrix(self.board.pose.q)
        long_axis = pad_rot[:, :, 1].unsqueeze(-1)
        in_board = (board_rot.transpose(-1, -2) @ long_axis).squeeze(-1)[:, :2]
        norm = torch.linalg.norm(in_board, dim=-1, keepdim=True)
        nominal = torch.zeros_like(in_board)
        nominal[:, 1] = 1.0
        along = torch.where(norm > 1e-6, in_board / norm.clamp(min=1e-9), nominal)
        across = torch.stack((-along[:, 1], along[:, 0]), dim=-1)
        return along, across

    def _face_on_board(self) -> torch.Tensor:
        board_up = quaternion_to_matrix(self.board.pose.q)[:, :, 2]
        face_normal = quaternion_to_matrix(self.pad.pose.q)[:, :, 0]
        square = (face_normal * board_up).sum(dim=-1) <= -self._face_square_cos
        height = self._pad_face_in_board()[:, 2]
        return square & ((height - self._face_half_thickness).abs() <= self._face_plane_tol)

    def _pad_in_hand(self) -> torch.Tensor:
        agent = cast("Panda", self.agent)
        return agent.is_grasping(self.pad) & (
            torch.linalg.norm(agent.tcp.pose.p - self.pad.pose.p, dim=-1) < self._pad_held_distance
        )

    def _pad_face_world(self) -> torch.Tensor:
        pad_axis = quaternion_to_matrix(self.pad.pose.q)[:, :, 0]
        return self.pad.pose.p + pad_axis * self._pad_face_centre_below_grip

    def _pad_face_in_board(self) -> torch.Tensor:
        board_rot = quaternion_to_matrix(self.board.pose.q)
        rel = (self._pad_face_world() - self.board.pose.p).unsqueeze(-1)
        return (board_rot.transpose(-1, -2) @ rel).squeeze(-1)

    def _pad_face_board_xy(self) -> torch.Tensor:
        return self._pad_face_in_board()[:, :2]

    def _pad_contact(self) -> tuple[torch.Tensor, torch.Tensor]:
        force = self.scene.get_pairwise_contact_forces(self.pad, self.board)
        board_up = quaternion_to_matrix(self.board.pose.q)[:, :, 2]
        along = (force * board_up).sum(dim=-1)
        tangential = torch.linalg.norm(force - along.unsqueeze(-1) * board_up, dim=-1)
        return along.abs(), tangential

    def _refresh_mark_decals(self) -> None:
        if not getattr(self, "_wipe_ready", False):
            return
        board_rot = quaternion_to_matrix(self.board.pose.q)
        board_p = self.board.pose.p
        board_q = self.board.pose.q
        parked = board_p.clone()
        parked[:, 2] = parked[:, 2] - 1.0

        frac = (self._dirt / self._dirt_budget.clamp(min=1e-9)).clamp(0.0, 1.0)
        worn = (self._mark_wear_tiers - torch.ceil(frac * self._mark_wear_tiers)).long()

        for i, (mx, my) in enumerate(self._mark_offsets):
            local = torch.tensor([mx, my, self._mark_proud], device=self.device, dtype=torch.float32)
            on_board = board_p + (board_rot @ local.expand(board_p.shape[0], 3).unsqueeze(-1)).squeeze(-1)
            for tier, decal in enumerate(self.mark_decals[i]):
                show = (worn[i] == tier).unsqueeze(-1)
                decal.set_pose(Pose.create_from_pq(torch.where(show, on_board, parked), board_q))

    @property
    def board_top_z(self) -> torch.Tensor:
        return self.board.pose.p[:, 2]

    def _markers_cleared(self) -> torch.Tensor:
        return (self._dirt <= 0.0).sum(dim=0) >= self._clear_count_required

    def _reset_wipe_trackers(self, env_idx: torch.Tensor, b: int) -> None:
        if not hasattr(self, "_dirt"):
            m = self._mark_count
            self._dirt = torch.zeros((m, self.num_envs), device=self.device)
            self._dirt_budget = torch.full((self.num_envs,), self._clear_path, device=self.device)
            self._pad_prev_local = torch.zeros((self.num_envs, 2), device=self.device)
            self._pad_prev_height = torch.zeros(self.num_envs, device=self.device)
            self._pad_prev_world = torch.zeros((self.num_envs, 3), device=self.device)
            self._passive_credit = torch.zeros((self._mark_count, self.num_envs), device=self.device)
            self._over_forced = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._over_force_run = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._board_downed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._wipe_sustain = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._mark_local = torch.as_tensor(
                np.asarray(self._mark_offsets, dtype=np.float32), device=self.device
            ).unsqueeze(1)
        budget = self._sample_clear_path(env_idx, b)
        self._dirt[:, env_idx] = budget.unsqueeze(0)
        self._dirt_budget[env_idx] = budget
        face = self._pad_face_in_board()
        self._pad_prev_local[env_idx] = face[env_idx, :2]
        self._pad_prev_height[env_idx] = face[env_idx, 2]
        self._pad_prev_world[env_idx] = self._pad_face_world()[env_idx]
        self._passive_credit[:, env_idx] = 0.0
        self._over_forced[env_idx] = False
        self._over_force_run[env_idx] = 0
        self._board_downed[env_idx] = False
        self._wipe_sustain[env_idx] = 0

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        board_pos = self.board.pose.p

        self._update_abrasion()
        self._refresh_mark_decals()

        cleared_count = (self._dirt <= 0.0).sum(dim=0)
        cleared = cleared_count >= self._clear_count_required

        held = self._pad_in_hand()

        board_top = self.board_top_z
        lifted = tcp_pos[:, 2] >= board_top + self._liftoff_tcp_clearance

        on_belt = (
            (board_pos[:, 2] > deck_z)
            & (torch.abs(board_pos[:, 0] - ox) < self._conveyor_half_width + self._board_half_x)
            & (torch.abs(board_pos[:, 1] - oy) < self._conveyor_half_length + self._board_half_y)
        )
        self._board_downed = self._board_downed | ~on_belt

        upright = self._board_upright()

        travelling = self._board_travelling(self._board_travel_frac, success_check=True)

        holding = cleared & held & lifted & on_belt & upright & travelling & ~self._over_forced & ~self._board_downed

        self._wipe_sustain = self._advance_sustain("wipe_sustain", self._wipe_sustain, holding)
        success = self._wipe_sustain >= self._wipe_sustain_steps

        press, tangential = self._pad_contact()
        return {
            "success": success,
            "marks_cleared": cleared,
            "cleared_count": cleared_count,
            "dirt_remaining": self._dirt.max(dim=0).values,
            "passive_credit": self._passive_credit.max(dim=0).values,
            "pad_press_N": press,
            "pad_tangential_N": tangential,
            "pad_face_on_board": self._face_on_board(),
            "pad_held": held,
            "pad_lifted_off": lifted,
            "board_on_belt": on_belt,
            "board_upright": upright,
            "board_travelling": travelling,
            "over_forced": self._over_forced,
            "board_downed": self._board_downed,
            "wipe_sustain": self._wipe_sustain,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p

        face_to_board = torch.linalg.norm(self._pad_face_world() - self.board.pose.p, dim=-1)
        reach_reward = 1 - torch.tanh(5 * face_to_board)

        wipe_reward = (1.0 - info["dirt_remaining"] / self._dirt_budget).clamp(0.0, 1.0)

        lift = (tcp_pos[:, 2] - self.board_top_z) / self._liftoff_tcp_clearance
        liftoff_reward = (
            lift.clamp(0.0, 1.0)
            * info["marks_cleared"].float()
            * info["pad_held"].float()
            * info["board_upright"].float()
            * info["board_travelling"].float()
        )

        failed = torch.maximum(info["over_forced"].float(), info["board_downed"].float())

        return reach_reward + 2.0 * wipe_reward + liftoff_reward - 2.0 * failed

    def compute_normalized_dense_reward(self, obs, action, info):
        return (self.compute_dense_reward(obs, action, info) + 2.0) / 6.0
