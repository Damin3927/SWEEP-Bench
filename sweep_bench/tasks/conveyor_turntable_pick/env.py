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
from sweep_bench.tasks.conveyor_base.on_belt import belt_carries
from sweep_bench.tasks.conveyor_turntable_pick.assets import (
    build_seated_part,
    build_turntable,
    build_turntable_plinth,
)
from sweep_bench.tasks.conveyor_turntable_pick.config import RandomizationConfig
from sweep_bench.tasks.conveyor_turntable_pick.scripted import ScriptedTurntablePickMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor

_TWO_PI = 2.0 * math.pi


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, _TWO_PI) - math.pi


@bench_task("ConveyorTurntablePick-v0")
@register_env("ConveyorTurntablePick-v0", max_episode_steps=300)
class ConveyorTurntablePickEnv(ScriptedTurntablePickMixin, ConveyorWorkspaceEnv):
    """Pick a part off a TURNTABLE that keeps rotating while the belt carries it."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _plinth_radius: float = 0.060
    _plinth_half_thickness: float = 0.004
    _boss_half_width: float = 0.026
    _well_half_width: float = 0.014
    _well_depth: float = 0.0045
    _ledge_thickness: float = 0.003
    _thrust_pad_thickness: float = 0.001
    _pin_clearance: float = 0.0005
    _plinth_density: float = 2000.0
    _plinth_patch_radius: float = 0.045

    _collar_radius: float = 0.012
    _collar_half_thickness: float = 0.002
    _pin_radius: float = 0.004
    _pin_half_length: float = 0.0025
    _disc_radius: float = 0.062
    _disc_half_thickness: float = 0.003
    _disc_density: float = 1000.0
    _journal_friction: float = 0.02

    _seat_radius: float = 0.038
    _part_half_width: float = 0.012
    _part_half_length: float = 0.019
    _part_half_height: float = 0.020
    _part_density: float = 1000.0
    _part_friction: float = 0.6
    _pocket_clearance: float = 0.0012
    _pocket_wall_thickness: float = 0.003
    _pocket_wall_height: float = 0.006

    _finger_pad_below_tcp: float = 0.0095

    _spin_rate_deg: float = 10.0
    _spin_gain: float = 0.014
    _spin_torque_max: float = 5.0e-3

    _spawn_phase_deg: tuple[float, float] = (-25.0, 25.0)
    _phase_wind_back_max: float = 4.0

    _grasp_distance_threshold: float = 0.06
    _carry_speed_match_threshold: float = 0.03
    _extract_clearance: float = 0.060
    _part_tilt_deg: float = 20.0
    _spin_keep_frac: float = 0.5
    _spin_stall_steps: int = 3
    _carrier_track_frac: float = 0.5
    _caught_moving_frac: float = 0.5
    _seat_offset_tol: float = 0.004
    _carrier_deck_drop: float = 0.005
    _carrier_on_belt_y: float = 0.84
    _carrier_on_belt_x: float = 0.055
    _pick_sustain_steps: int = 5

    default_prompt: str = "pick the block off the turning table on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _aperture_half_width(self) -> float:
        return self._pin_radius + self._pin_clearance

    @property
    def _well_floor_z(self) -> float:
        return 2.0 * self._plinth_half_thickness + self._thrust_pad_thickness

    @property
    def _hub_rest_z(self) -> float:
        return self._well_floor_z + self._collar_half_thickness

    @property
    def _ledge_top_local(self) -> float:
        return self._well_floor_z + self._well_depth + self._ledge_thickness

    @property
    def _plate_bottom_local(self) -> float:
        return self._ledge_top_local - self._hub_rest_z + 0.0015

    @property
    def _plate_top_local(self) -> float:
        return self._plate_bottom_local + 2.0 * self._disc_half_thickness

    @property
    def _pocket_floor_z(self) -> float:
        return self._hub_rest_z + self._plate_top_local

    @property
    def _part_rest_z(self) -> float:
        return self._pocket_floor_z + self._part_half_height

    @property
    def _pocket_half_x(self) -> float:
        return self._part_half_width + self._pocket_clearance

    @property
    def _pocket_half_y(self) -> float:
        return self._part_half_length + self._pocket_clearance

    @property
    def _spin_rate(self) -> float:
        return math.radians(self._spin_rate_deg)

    def part_seat_offset(self) -> torch.Tensor:
        local = (self.turntable.pose.inv() * self.part.pose).p
        centre = torch.zeros_like(local[:, :2])
        centre[:, 0] = self._seat_radius
        return local[:, :2] - centre

    def part_in_pocket(self) -> torch.Tensor:
        offset = self.part_seat_offset().abs()
        return (offset[:, 0] <= self._seat_offset_tol) & (offset[:, 1] <= self._seat_offset_tol)

    def disc_spin_command(self) -> torch.Tensor:
        return self._spin_sign * self._spin_rate

    def disc_phase(self) -> torch.Tensor:
        rot_c = quaternion_to_matrix(self.plinth.pose.q)
        rot_d = quaternion_to_matrix(self.turntable.pose.q)
        rel = torch.bmm(rot_c.transpose(1, 2), rot_d)
        return torch.atan2(rel[:, 1, 0], rel[:, 0, 0])

    def disc_spin(self) -> torch.Tensor:
        axis = quaternion_to_matrix(self.plinth.pose.q)[:, :, 2]
        return ((self.turntable.angular_velocity - self.plinth.angular_velocity) * axis).sum(dim=-1)

    def part_axis_xy(self) -> torch.Tensor:
        axis = quaternion_to_matrix(self.part.pose.q)[:, :2, 1]
        return axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=1e-9)

    def disc_rate(self) -> torch.Tensor:
        return _wrap(self.disc_phase() - self._prev_phase) * self.control_freq

    def part_lift_above_seat(self) -> torch.Tensor:
        local = (self.plinth.pose.inv() * self.part.pose).p
        return local[:, 2] - self._part_rest_z

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        plinth_builder = build_turntable_plinth(
            self.scene,
            plinth_radius=self._plinth_radius,
            plinth_half_thickness=self._plinth_half_thickness,
            boss_half_width=self._boss_half_width,
            well_half_width=self._well_half_width,
            well_depth=self._well_depth,
            ledge_thickness=self._ledge_thickness,
            aperture_half_width=self._aperture_half_width,
            thrust_pad_thickness=self._thrust_pad_thickness,
            density=self._plinth_density,
            journal_friction=self._journal_friction,
        )
        plinth_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z])
        self.plinth = plinth_builder.build(name="turntable_plinth")
        self.belt.register_rider(self.plinth, patch_radius=self._plinth_patch_radius)

        table_builder = build_turntable(
            self.scene,
            collar_radius=self._collar_radius,
            collar_half_thickness=self._collar_half_thickness,
            pin_radius=self._pin_radius,
            pin_half_length=self._pin_half_length,
            disc_radius=self._disc_radius,
            disc_half_thickness=self._disc_half_thickness,
            plate_bottom_z=self._plate_bottom_local,
            seat_radius=self._seat_radius,
            pocket_half_x=self._pocket_half_x,
            pocket_half_y=self._pocket_half_y,
            wall_thickness=self._pocket_wall_thickness,
            wall_height=self._pocket_wall_height,
            density=self._disc_density,
            friction=self._journal_friction,
        )
        table_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._hub_rest_z])
        self.turntable = table_builder.build(name="turntable")

        part_builder = build_seated_part(
            self.scene,
            half_width=self._part_half_width,
            half_length=self._part_half_length,
            half_height=self._part_half_height,
            density=self._part_density,
            friction=self._part_friction,
        )
        part_builder.initial_pose = sapien.Pose(p=[ox + self._seat_radius, oy, deck_z + self._part_rest_z])
        self.part = part_builder.build(name="turntable_part")

    @property
    def _task_actors(self) -> list[Actor]:
        return [self.part, *self.belt.riders, self.turntable]

    @property
    def _primary_rider(self) -> Actor:
        return self.part

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        sign_np = np.ones(b, dtype=np.float32)
        if self.randomization.spin_direction:
            sign_np = self._episode_rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=b)
        sign = torch.as_tensor(sign_np, device=device)

        lo, hi = self._spawn_phase_deg
        phase_np = np.radians(self._episode_rng.uniform(lo, hi, size=b)).astype(np.float32)
        phase = torch.as_tensor(phase_np, device=device)
        phase = phase + torch.where(
            self.belt.direction_sign[env_idx] < 0.0, torch.full_like(phase, math.pi), torch.zeros_like(phase)
        )

        plinth_pos = self._sample_plinth_position(env_idx, b)
        phase = phase - sign * self._spin_rate * self._phase_wind_back_time(env_idx, plinth_pos)

        plinth_q = torch.zeros((b, 4), device=device)
        plinth_q[:, 0] = 1.0
        self.plinth.set_pose(Pose.create_from_pq(plinth_pos, plinth_q))
        belt_vel = torch.zeros((b, 3), device=device)
        belt_vel[:, 1] = self.belt.velocity[env_idx]
        self.plinth.set_linear_velocity(belt_vel)
        self.plinth.set_angular_velocity(torch.zeros((b, 3), device=device))

        omega = sign * self._spin_rate
        table_q = torch.zeros((b, 4), device=device)
        table_q[:, 0] = torch.cos(phase * 0.5)
        table_q[:, 3] = torch.sin(phase * 0.5)
        table_pos = plinth_pos.clone()
        table_pos[:, 2] = table_pos[:, 2] + self._hub_rest_z
        self.turntable.set_pose(Pose.create_from_pq(table_pos, table_q))
        self.turntable.set_linear_velocity(belt_vel)
        table_omega = torch.zeros((b, 3), device=device)
        table_omega[:, 2] = omega
        self.turntable.set_angular_velocity(table_omega)

        offset = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1) * self._seat_radius
        part_pos = plinth_pos.clone()
        part_pos[:, :2] = part_pos[:, :2] + offset
        part_pos[:, 2] = plinth_pos[:, 2] + self._part_rest_z
        self.part.set_pose(Pose.create_from_pq(part_pos, table_q))
        part_vel = belt_vel.clone()
        part_vel[:, 0] = part_vel[:, 0] - omega * offset[:, 1]
        part_vel[:, 1] = part_vel[:, 1] + omega * offset[:, 0]
        self.part.set_linear_velocity(part_vel)
        self.part.set_angular_velocity(table_omega)

        self._reset_pick_trackers(env_idx)
        self._spin_sign[env_idx] = sign
        self._spin_ready = True

    def _phase_wind_back_time(self, env_idx: torch.Tensor, plinth_pos: torch.Tensor) -> torch.Tensor:
        oy = self.workspace_offset[1]
        speed = torch.abs(self.belt.velocity[env_idx])
        travel = torch.abs(plinth_pos[:, 1] - oy)
        return torch.where(speed > 1e-6, travel / speed.clamp(min=1e-6), torch.zeros_like(travel)).clamp(
            max=self._phase_wind_back_max
        )

    def _sample_plinth_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        device = self.device
        start_y = self._conveyor_half_length - self._plinth_radius - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.lane_offset:
            lo, hi = self.randomization.lane_offset_range
            draws = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = ox + torch.as_tensor(draws, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = oz + self._conveyor_deck_height
        return pos

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_spin_drive()

    def _apply_spin_drive(self) -> None:
        if not getattr(self, "_spin_ready", False):
            return
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()
        axis = quaternion_to_matrix(self.plinth.pose.q)[:, :, 2]
        err = self.disc_spin_command() - self.disc_spin()
        tau = (self._spin_gain * err).clamp(-self._spin_torque_max, self._spin_torque_max)
        torque = axis * tau.unsqueeze(-1)
        if gpu:
            torque_buf = self.scene.px.cuda_rigid_body_torque.torch()
            torque_buf[self.turntable._body_data_index, :3] = torque
            self.scene.px.gpu_apply_rigid_dynamic_torque()
        else:
            self.turntable._bodies[0].add_force_torque(np.zeros(3, dtype=np.float32), torque.squeeze(0).cpu().numpy())

    def _reset_pick_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_pick_sustain"):
            b, device = self.num_envs, self.device
            self._pick_sustain = torch.zeros(b, dtype=torch.long, device=device)
            self._spin_sign = torch.ones(b, device=device)
            self._spin_kept = torch.ones(b, dtype=torch.bool, device=device)
            self._spin_stall = torch.zeros(b, dtype=torch.long, device=device)
            self._was_gripping = torch.zeros(b, dtype=torch.bool, device=device)
            self._pre_grasp_carried = torch.ones(b, dtype=torch.bool, device=device)
            self._caught_moving = torch.ones(b, dtype=torch.bool, device=device)
            self._seated_at_grasp = torch.zeros(b, dtype=torch.bool, device=device)
            self._prev_part_pos = torch.zeros((b, 3), device=device)
            self._prev_tcp_pos = torch.zeros((b, 3), device=device)
            self._prev_phase = torch.zeros(b, device=device)
        self._pick_sustain[env_idx] = 0
        self._spin_kept[env_idx] = True
        self._spin_stall[env_idx] = 0
        self._was_gripping[env_idx] = False
        self._pre_grasp_carried[env_idx] = True
        self._caught_moving[env_idx] = True
        self._seated_at_grasp[env_idx] = False
        self._prev_phase[env_idx] = self.disc_phase()[env_idx]
        self._prev_part_pos[env_idx] = self.part.pose.p[env_idx]
        self._prev_tcp_pos[env_idx] = cast("Panda", self.agent).tcp.pose.p[env_idx]

    @property
    def _belt_push_watch(self) -> Actor:
        return self.plinth

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        part_pos = self.part.pose.p
        plinth_pos = self.plinth.pose.p
        advanced = self._eval_advanced("turntable_pick_step")

        gripping = agent.is_grasping(self.part)
        co_located = torch.linalg.norm(tcp_pos - part_pos, dim=-1) < self._grasp_distance_threshold
        rel = (part_pos - tcp_pos) - (self._prev_part_pos - self._prev_tcp_pos)
        rel_speed = torch.linalg.norm(rel, dim=-1) * self.control_freq
        self._prev_part_pos = torch.where(advanced.unsqueeze(-1), part_pos, self._prev_part_pos)
        self._prev_tcp_pos = torch.where(advanced.unsqueeze(-1), tcp_pos, self._prev_tcp_pos)
        carried = rel_speed < self._carry_speed_match_threshold
        is_held = gripping & co_located & carried

        lift = self.part_lift_above_seat()
        extracted = lift >= self._extract_clearance

        upright = quaternion_to_matrix(self.part.pose.q)[:, 2, 2] >= math.cos(math.radians(self._part_tilt_deg))

        spin = self.disc_spin()
        phase = self.disc_phase()
        rate = _wrap(phase - self._prev_phase) * self.control_freq
        self._prev_phase = torch.where(advanced, phase, self._prev_phase)
        command = self.disc_spin_command()
        turning = rate * command >= self._spin_keep_frac * command**2
        watching = ~gripping
        step = self._eval_advanced("turntable_stall")
        cleared = watching & turning & step
        self._spin_stall = torch.where(
            cleared,
            torch.zeros_like(self._spin_stall),
            self._spin_stall + (watching & ~turning & step).long(),
        )
        self._spin_kept = self._spin_kept & (self._spin_stall < self._spin_stall_steps)

        on_belt = (
            (torch.abs(plinth_pos[:, 2] - deck_z) < self._carrier_deck_drop)
            & (torch.abs(plinth_pos[:, 0] - ox) < self._carrier_on_belt_x)
            & (torch.abs(plinth_pos[:, 1] - oy) < self._carrier_on_belt_y)
        )
        belt_v = self.belt.velocity
        travelling = belt_carries(self.plinth.linear_velocity[:, 1], belt_v, self._carrier_track_frac)
        riding = on_belt & travelling

        carried_by_belt = (
            self.belt_carried(self.plinth, self._caught_moving_frac)
            & self.riding_on_deck(self.plinth, 0.0)
            & ~self.gripper_drove_along_belt()
        )
        self._pre_grasp_carried = torch.where(advanced & ~gripping, carried_by_belt, self._pre_grasp_carried)
        first_grasp = gripping & ~self._was_gripping & advanced
        self._caught_moving = torch.where(first_grasp, self._pre_grasp_carried, self._caught_moving)
        self._seated_at_grasp = torch.where(
            first_grasp,
            (lift <= self._pocket_wall_height) & self.part_in_pocket(),
            self._seated_at_grasp,
        )
        self._spin_kept = self._spin_kept & ~(first_grasp & (self._spin_stall > 0))
        self._was_gripping = torch.where(advanced, gripping, self._was_gripping)

        picked = is_held & extracted & upright & self._spin_kept & riding & self._seated_at_grasp & self._caught_moving
        self._pick_sustain = self._advance_sustain("turntable_pick", self._pick_sustain, picked)
        success = self._pick_sustain >= self._pick_sustain_steps

        return {
            "success": success,
            "is_held": is_held,
            "extracted": extracted,
            "upright": upright,
            "spin_kept": self._spin_kept,
            "seated_at_grasp": self._seated_at_grasp,
            "caught_moving": self._caught_moving,
            "riding": riding,
            "disc_phase": phase,
            "disc_spin": spin,
            "disc_rate": rate,
            "table_turning": turning,
            "spin_stall": self._spin_stall,
            "part_lift": lift,
            "part_seat_offset": self.part_seat_offset(),
            "part_in_pocket": self.part_in_pocket(),
            "seat_tangential_speed": torch.abs(self.disc_spin_command()) * self._seat_radius,
            "pick_sustain": self._pick_sustain,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        part_pos = self.part.pose.p

        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - part_pos, dim=-1))
        lift_reward = torch.clamp(info["part_lift"] / self._extract_clearance, 0, 1)
        credit = (
            info["is_held"].float()
            * info["spin_kept"].float()
            * info["upright"].float()
            * info["riding"].float()
            * info["seated_at_grasp"].float()
            * info["caught_moving"].float()
        )
        return reach_reward + 2.0 * lift_reward * credit

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
