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
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.conveyor_drive import apply_drive_writes
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import belt_carries
from sweep_bench.tasks.conveyor_nut_rundown.assets import build_nut_fixture, build_wing_nut
from sweep_bench.tasks.conveyor_nut_rundown.config import RandomizationConfig
from sweep_bench.tasks.conveyor_nut_rundown.scripted import ScriptedNutRunDownMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor

_TWO_PI = 2.0 * math.pi


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, _TWO_PI) - math.pi


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    t = x.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@bench_task("ConveyorNutRunDown-v0")
@register_env("ConveyorNutRunDown-v0", max_episode_steps=300)
class ConveyorNutRunDownEnv(ScriptedNutRunDownMixin, ConveyorWorkspaceEnv):
    """Run a wing nut down a threaded post until it seats, on a moving belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _plate_half_x: float = 0.036
    _plate_half_y: float = 0.050
    _plate_height: float = 0.020
    _boss_radius: float = 0.020
    _boss_height: float = 0.008
    _post_radius: float = 0.0065
    _post_height: float = 0.040
    _fixture_density: float = 11000.0

    _bore_clearance: float = 0.0006
    _nut_body_half: float = 0.013
    _nut_half_height: float = 0.010
    _wing_count: int = 2
    _wing_outer_radius: float = 0.050
    _wing_half_thickness: float = 0.0035
    _wing_half_height: float = 0.010
    _wing_contact_radius: float = 0.040
    _nut_density: float = 7800.0
    _nut_friction: float = 0.4

    _patch_radius: float = 0.030

    _patch_yaw_stiffness: float = 2.0
    _patch_yaw_cone_frac: float = 1.0

    _thread_lead: float = 0.023
    _rundown_travel: float = 0.0035
    _preload_turn: float = math.radians(8.0)
    _thread_stiffness: float = 600.0
    _thread_damping: float = 2.0
    _thread_force_max: float = 8.0
    _preload_force: float = 2.5
    _preload_span: float = math.radians(8.0)
    _thread_torque: float = 0.0015
    _thread_viscous: float = 2.0e-4
    _thread_torque_max: float = 0.01
    _thread_cap_frac: float = 0.8
    _coupling_band: float = 0.0010
    _nut_upright_cos: float = math.cos(math.radians(12.0))
    _bore_lateral_tol: float = 0.0025
    _engage_depth: float = 0.006

    _seat_gap_tol: float = 0.0003
    _seat_force_min: float = 0.8
    _nut_tcp_clear: float = 0.045
    _fixture_deck_drop: float = 0.005
    _fixture_on_belt_y: float = 0.85
    _fixture_tilt_cos: float = math.cos(math.radians(10.0))
    _body_travel_frac: float = 0.5
    _rundown_sustain_steps: int = 8
    _lost_lateral: float = 0.0075
    _alias_guard: float = math.radians(30.0)

    default_prompt: str = "screw the wing nut down the post until it seats on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _bore_half_width(self) -> float:
        return self._post_radius + self._bore_clearance

    @property
    def _seat_plane_z(self) -> float:
        return self._plate_height + self._boss_height

    @property
    def _post_top_z(self) -> float:
        return self._seat_plane_z + self._post_height

    @property
    def _nut_seat_h(self) -> float:
        return self._seat_plane_z + self._nut_half_height

    @property
    def _nut_outer_radius(self) -> float:
        return math.hypot(self._wing_outer_radius, self._wing_half_thickness)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        fixture_builder = build_nut_fixture(
            self.scene,
            plate_half_x=self._plate_half_x,
            plate_half_y=self._plate_half_y,
            plate_height=self._plate_height,
            boss_radius=self._boss_radius,
            boss_height=self._boss_height,
            post_radius=self._post_radius,
            post_height=self._post_height,
            density=self._fixture_density,
        )
        fixture_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z])
        self.fixture = fixture_builder.build(name="nut_fixture")
        self.belt.register_rider(self.fixture, patch_radius=self._patch_radius)

        nut_builder = build_wing_nut(
            self.scene,
            bore_half_width=self._bore_half_width,
            body_half_width=self._nut_body_half,
            body_half_height=self._nut_half_height,
            wing_count=self._wing_count,
            wing_outer_radius=self._wing_outer_radius,
            wing_half_thickness=self._wing_half_thickness,
            wing_half_height=self._wing_half_height,
            density=self._nut_density,
            friction=self._nut_friction,
        )
        nut_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._nut_seat_h])
        self.nut = nut_builder.build(name="wing_nut")

        self._nut_mass = float(self.nut.mass[0])
        self._nut_inertia = float(max(self.nut._bodies[0].inertia))
        self._fixture_weight = 9.81 * float(self.fixture.mass[0])

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.nut]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        fixture_q = self._sample_fixture_orientation(env_idx, b)
        fixture_pos = self._sample_fixture_position(env_idx, b)
        self.fixture.set_pose(Pose.create_from_pq(fixture_pos, fixture_q))
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        self.fixture.set_linear_velocity(vel)
        self.fixture.set_angular_velocity(torch.zeros((b, 3), device=device))

        start_angle = self._sample_nut_start_angle(b)
        start_gap = self._sample_nut_start_gap(b)
        nut_local = torch.zeros((b, 3), device=device)
        nut_local[:, 2] = self._nut_seat_h + start_gap
        rot = quaternion_to_matrix(fixture_q)
        nut_pos = fixture_pos + torch.bmm(rot, nut_local.unsqueeze(-1)).squeeze(-1)
        spin_q = torch.zeros((b, 4), device=device)
        spin_q[:, 0] = torch.cos(start_angle * 0.5)
        spin_q[:, 3] = torch.sin(start_angle * 0.5)
        self.nut.set_pose(Pose.create_from_pq(nut_pos, quaternion_multiply(fixture_q, spin_q)))
        self.nut.set_linear_velocity(vel)
        self.nut.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_rundown_trackers(env_idx)
        self._nut_yaw_prev[env_idx] = start_angle
        self._nut_start_h[env_idx] = self._nut_seat_h + start_gap
        self._turn_to_seat[env_idx] = start_gap * _TWO_PI / self._thread_lead
        self._patch_yaw_ref[env_idx] = self._fixture_yaw()[env_idx]
        self._thread_ready = True

    def _sample_fixture_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        yaw_np = self._episode_rng.uniform(*self.randomization.yaw_range, size=b).astype(np.float32)
        yaw = (
            torch.as_tensor(yaw_np, device=device) if self.randomization.fixture_yaw else torch.zeros(b, device=device)
        )
        yaw = yaw + torch.where(
            self.belt.direction_sign[env_idx] < 0.0, torch.full_like(yaw, math.pi), torch.zeros_like(yaw)
        )
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = torch.cos(yaw * 0.5)
        q[:, 3] = torch.sin(yaw * 0.5)
        return q

    def _sample_nut_start_angle(self, b: int) -> torch.Tensor:
        if not self.randomization.nut_start_angle:
            return torch.zeros(b, device=self.device)
        lo, hi = self.randomization.start_angle_range
        return torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)

    def _sample_nut_start_gap(self, b: int) -> torch.Tensor:
        if not self.randomization.nut_start_height:
            return torch.full((b,), self._rundown_travel, device=self.device)
        lo, hi = self.randomization.start_height_range
        return torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)

    def _sample_fixture_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._plate_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.rider_lateral_offset:
            lo, hi = self.randomization.lateral_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = pos[:, 0] + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z
        return pos

    def _before_simulation_step(self) -> None:
        writes = self.belt.drive.compute_writes()
        reaction = self._apply_thread_law()
        writes = self._with_patch_yaw_friction(writes)
        apply_drive_writes(self.scene, self._with_thread_reaction(writes, reaction))

    def _with_patch_yaw_friction(
        self, writes: list[tuple[Actor, torch.Tensor, torch.Tensor]]
    ) -> list[tuple[Actor, torch.Tensor, torch.Tensor]]:
        if not writes or not getattr(self, "_thread_ready", False):
            return writes
        yaw = self._fixture_yaw()
        cap = self._patch_yaw_cone(self._fixture_normal_force())
        slack = (cap / self._patch_yaw_stiffness).clamp(min=1e-9)
        err = _wrap(self._patch_yaw_ref - yaw).clamp(-slack, slack)
        self._patch_yaw_ref = yaw + err
        tau = self._patch_yaw_stiffness * err
        out = []
        for actor, force, torque in writes:
            if actor is self.fixture:
                torque = torque.clone()
                torque[:, 2] = torque[:, 2] + tau
            out.append((actor, force, torque))
        return out

    def _fixture_normal_force(self) -> torch.Tensor:
        contact = self.scene.get_pairwise_contact_forces(self.fixture, self.belt.actor)
        return contact[:, 2].clamp(min=0.0, max=5.0 * self._fixture_weight)

    def _patch_yaw_cone(self, normal: torch.Tensor) -> torch.Tensor:
        return self._patch_yaw_cone_frac * (2.0 / 3.0) * self.belt_static_friction * normal * self._patch_radius

    def _with_thread_reaction(
        self,
        writes: list[tuple[Actor, torch.Tensor, torch.Tensor]],
        reaction: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> list[tuple[Actor, torch.Tensor, torch.Tensor]]:
        if reaction is None:
            return writes
        r_force, r_torque = reaction
        out = []
        for actor, force, torque in writes:
            if actor is self.fixture:
                force, torque = force + r_force, torque + r_torque
            out.append((actor, force, torque))
        return out

    def _thread_reaction(self, force: torch.Tensor, torque: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lever = self.nut.pose.p - self.fixture.pose.p
        return -force, -torque - torch.linalg.cross(lever, force)

    def _apply_thread_law(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not getattr(self, "_thread_ready", False):
            return None
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()

        up = quaternion_to_matrix(self.fixture.pose.q)[:, :, 2]
        height = self._nut_height()
        yaw = self._nut_yaw()
        step = _wrap(yaw - self._nut_yaw_prev)
        self._nut_spun = self._nut_spun | (step.abs() > self._alias_guard)
        self._nut_yaw_prev = yaw

        engaged = self._thread_engaged(height)
        self._latch_nut_lost(self._nut_lateral(), self._bore_wrap(height))
        error = height - self._thread_height(self._rundown_turn)
        error_post = height - self._thread_height(self._rundown_turn + step)
        followable = (step <= 0.0) | (error_post.abs() <= self._coupling_band)
        coupled = engaged & (error.abs() <= self._coupling_band) & followable
        self._coupling_worst = torch.maximum(self._coupling_worst, error.abs())

        self._rundown_turn = self._rundown_turn + step * coupled

        target = self._thread_height(self._rundown_turn)
        rel_v = ((self.nut.linear_velocity - self.fixture.linear_velocity) * up).sum(dim=-1)
        omega = self._nut_spin()
        v_thread = torch.where(
            target > self._nut_seat_h,
            -self._thread_lead * omega / _TWO_PI,
            torch.zeros_like(omega),
        )
        preload = self._preload_force * _smoothstep((self._rundown_turn - self._turn_to_seat) / self._preload_span)
        axial = (
            9.81 * self._nut_mass
            + self._thread_stiffness * (target - height)
            - self._thread_damping * (rel_v - v_thread)
            - preload
        ).clamp(-self._thread_force_max, self._thread_force_max)
        axial = axial * engaged
        force = up * axial.unsqueeze(-1)

        dt = 1.0 / self.sim_config.sim_freq
        cap = self._thread_cap_frac * self._nut_inertia * omega.abs() / dt
        tau = (
            -torch.sign(omega) * torch.minimum(torch.full_like(cap, self._thread_torque), cap)
            - self._thread_viscous * omega
        ).clamp(-self._thread_torque_max, self._thread_torque_max)
        torque = up * (tau * engaged).unsqueeze(-1)

        if gpu:
            px = self.scene.px
            px.cuda_rigid_body_force.torch()[self.nut._body_data_index, :3] = force
            px.cuda_rigid_body_torque.torch()[self.nut._body_data_index, :3] = torque
            px.gpu_apply_rigid_dynamic_force()
            px.gpu_apply_rigid_dynamic_torque()
        else:
            self.nut._bodies[0].add_force_torque(force.squeeze(0).cpu().numpy(), torque.squeeze(0).cpu().numpy())
        return self._thread_reaction(force, torque)

    def _fixture_yaw(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.fixture.pose.q)
        return torch.atan2(rot[:, 1, 0], rot[:, 0, 0])

    def _nut_local(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.fixture.pose.q)
        return torch.bmm(rot.transpose(1, 2), (self.nut.pose.p - self.fixture.pose.p).unsqueeze(-1)).squeeze(-1)

    def _nut_height(self) -> torch.Tensor:
        return self._nut_local()[:, 2]

    def _nut_lateral(self) -> torch.Tensor:
        return torch.linalg.norm(self._nut_local()[:, :2], dim=-1)

    def _nut_yaw(self) -> torch.Tensor:
        rot_f = quaternion_to_matrix(self.fixture.pose.q)
        rot_n = quaternion_to_matrix(self.nut.pose.q)
        rel = torch.bmm(rot_f.transpose(1, 2), rot_n)
        return torch.atan2(rel[:, 1, 0], rel[:, 0, 0])

    def _nut_spin(self) -> torch.Tensor:
        up = quaternion_to_matrix(self.fixture.pose.q)[:, :, 2]
        return ((self.nut.angular_velocity - self.fixture.angular_velocity) * up).sum(dim=-1)

    def _nut_upright(self) -> torch.Tensor:
        up_f = quaternion_to_matrix(self.fixture.pose.q)[:, :, 2]
        up_n = quaternion_to_matrix(self.nut.pose.q)[:, :, 2]
        return (up_f * up_n).sum(dim=-1)

    def _bore_wrap(self, height: torch.Tensor) -> torch.Tensor:
        top = torch.minimum(height + self._nut_half_height, torch.full_like(height, self._post_top_z))
        bottom = torch.maximum(height - self._nut_half_height, torch.full_like(height, self._seat_plane_z))
        return (top - bottom).clamp(min=0.0)

    def _latch_nut_lost(self, lateral: torch.Tensor, wrap: torch.Tensor) -> None:
        self._nut_lost = self._nut_lost | (lateral > self._lost_lateral) | (wrap <= 0.0)

    def _thread_engaged(self, height: torch.Tensor) -> torch.Tensor:
        return (
            (self._nut_upright() >= self._nut_upright_cos)
            & (self._nut_lateral() <= self._bore_lateral_tol)
            & (self._bore_wrap(height) >= self._engage_depth)
        )

    def _thread_height(self, turn: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            self._nut_start_h - self._thread_lead * turn / _TWO_PI,
            min=self._nut_seat_h,
        )

    def _seat_gap(self, height: torch.Tensor) -> torch.Tensor:
        rot_f = quaternion_to_matrix(self.fixture.pose.q)
        rot_n = quaternion_to_matrix(self.nut.pose.q)
        up_in_nut = torch.bmm(rot_n.transpose(1, 2), rot_f[:, :, 2].unsqueeze(-1)).squeeze(-1)
        horizontal = up_in_nut[:, :2]
        sin_tilt = torch.linalg.norm(horizontal, dim=-1)
        cos_tilt = up_in_nut[:, 2].clamp(-1.0, 1.0)
        down = -horizontal / sin_tilt.clamp(min=1e-9).unsqueeze(-1)
        drop = self._nut_half_height * cos_tilt + self._underside_support(down) * sin_tilt
        return height - drop - self._seat_plane_z

    def _underside_support(self, down: torch.Tensor) -> torch.Tensor:
        square = self._nut_body_half * (down[:, 0].abs() + down[:, 1].abs())
        rim = min(self._boss_radius, self._nut_outer_radius)
        support = square
        for k in range(self._wing_count):
            phi = k * _TWO_PI / self._wing_count
            along = -math.sin(phi) * down[:, 0] + math.cos(phi) * down[:, 1]
            across = math.cos(phi) * down[:, 0] + math.sin(phi) * down[:, 1]
            reach = torch.where(along >= 0.0, rim * along, self._nut_body_half * along)
            support = torch.maximum(support, reach + self._wing_half_thickness * across.abs())
        return support

    def _seat_force(self) -> torch.Tensor:
        up = quaternion_to_matrix(self.fixture.pose.q)[:, :, 2]
        return (self.scene.get_pairwise_contact_forces(self.nut, self.fixture) * up).sum(dim=-1)

    def _nut_clearance(self) -> torch.Tensor:
        agent = cast("Panda", self.agent)
        rot_nut = quaternion_to_matrix(self.nut.pose.q)
        local = torch.bmm(rot_nut.transpose(1, 2), (agent.tcp.pose.p - self.nut.pose.p).unsqueeze(-1)).squeeze(-1)
        radial = (torch.linalg.norm(local[:, :2], dim=-1) - self._nut_outer_radius).clamp(min=0.0)
        axial = (local[:, 2] - self._nut_half_height).clamp(min=0.0) + (-self._nut_half_height - local[:, 2]).clamp(
            min=0.0
        )
        return torch.sqrt(radial * radial + axial * axial)

    def _wing_distance(self) -> torch.Tensor:
        agent = cast("Panda", self.agent)
        rot = quaternion_to_matrix(self.fixture.pose.q)
        local = torch.bmm(rot.transpose(1, 2), (agent.tcp.pose.p - self.fixture.pose.p).unsqueeze(-1)).squeeze(-1)
        yaw = self._nut_yaw()
        best = None
        for k in range(self._wing_count):
            phi = yaw + k * _TWO_PI / self._wing_count
            point = torch.stack(
                [
                    -self._wing_contact_radius * torch.sin(phi),
                    self._wing_contact_radius * torch.cos(phi),
                    self._nut_height(),
                ],
                dim=-1,
            )
            d = torch.linalg.norm(local - point, dim=-1)
            best = d if best is None else torch.minimum(best, d)
        assert best is not None
        return best

    @property
    def _turn_target(self) -> torch.Tensor:
        return self._turn_to_seat + self._preload_turn

    def _reset_rundown_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_rundown_sustain"):
            b, device = self.num_envs, self.device
            self._rundown_sustain = torch.zeros(b, dtype=torch.long, device=device)
            self._rundown_turn = torch.zeros(b, device=device)
            self._nut_yaw_prev = torch.zeros(b, device=device)
            self._nut_start_h = torch.full((b,), self._nut_seat_h + self._rundown_travel, device=device)
            self._turn_to_seat = torch.full((b,), self._rundown_travel * _TWO_PI / self._thread_lead, device=device)
            self._coupling_worst = torch.zeros(b, device=device)
            self._patch_yaw_ref = torch.zeros(b, device=device)
            self._nut_lost = torch.zeros(b, dtype=torch.bool, device=device)
            self._nut_spun = torch.zeros(b, dtype=torch.bool, device=device)
        self._rundown_sustain[env_idx] = 0
        self._rundown_turn[env_idx] = 0.0
        self._coupling_worst[env_idx] = 0.0
        self._nut_lost[env_idx] = False
        self._nut_spun[env_idx] = False

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        turn = self._rundown_turn
        height = self._nut_height()
        fixture_pos = self.fixture.pose.p

        run_down = turn >= self._turn_target

        gap = self._seat_gap(height)
        seated = gap <= self._seat_gap_tol
        seat_force = self._seat_force()
        preloaded = seat_force >= self._seat_force_min

        error = height - self._thread_height(turn)
        coupled = error.abs() <= self._coupling_band

        upright = self._nut_upright()
        lateral = self._nut_lateral()
        wrap = self._bore_wrap(height)
        on_post = (
            (upright >= self._nut_upright_cos) & (lateral <= self._bore_lateral_tol) & (wrap >= self._engage_depth)
        )
        self._latch_nut_lost(lateral, wrap)

        nut_dist = self._nut_clearance()
        arm_clear = nut_dist >= self._nut_tcp_clear

        fixture_upright = quaternion_to_matrix(self.fixture.pose.q)[:, 2, 2] >= self._fixture_tilt_cos
        on_belt = (
            (torch.abs(fixture_pos[:, 2] - deck_z) < self._fixture_deck_drop)
            & (torch.abs(fixture_pos[:, 0] - ox) < self._conveyor_half_width)
            & (torch.abs(fixture_pos[:, 1] - oy) < self._fixture_on_belt_y)
        )

        belt_v = self.belt.velocity
        travelling = belt_carries(self.fixture.linear_velocity[:, 1], belt_v, self._body_travel_frac)

        failed = self._nut_lost | self._nut_spun

        holding = (
            run_down
            & seated
            & preloaded
            & coupled
            & on_post
            & arm_clear
            & on_belt
            & fixture_upright
            & travelling
            & ~failed
        )
        self._rundown_sustain = self._advance_sustain("rundown_sustain", self._rundown_sustain, holding)
        success = self._rundown_sustain >= self._rundown_sustain_steps

        return {
            "success": success,
            "rundown_turn": turn,
            "turn_target": self._turn_target,
            "run_down": run_down,
            "seat_gap": gap,
            "seated": seated,
            "seat_force_N": seat_force,
            "seat_preloaded": preloaded,
            "coupling_error": error,
            "coupling_worst": self._coupling_worst,
            "thread_coupled": coupled,
            "nut_upright": upright,
            "nut_lateral": lateral,
            "bore_wrap": wrap,
            "nut_on_post": on_post,
            "tcp_to_nut": nut_dist,
            "arm_clear": arm_clear,
            "fixture_on_belt": on_belt,
            "fixture_upright": fixture_upright,
            "fixture_travelling": travelling,
            "nut_lost": self._nut_lost,
            "nut_spun": self._nut_spun,
            "rundown_sustain": self._rundown_sustain,
        }

    def compute_dense_reward(self, obs, action, info):
        reach_reward = 1 - torch.tanh(5 * self._wing_distance())

        progress = (info["rundown_turn"] / self._turn_target).clamp(0.0, 1.0)

        finishable = (
            info["nut_on_post"]
            & info["thread_coupled"]
            & info["fixture_on_belt"]
            & info["fixture_upright"]
            & info["fixture_travelling"]
        ).to(progress.dtype)

        preload_reward = (
            (info["seat_force_N"] / self._seat_force_min).clamp(0.0, 1.0)
            * info["seated"].to(progress.dtype)
            * finishable
        )

        withdraw_reward = (
            info["run_down"].to(progress.dtype)
            * (info["tcp_to_nut"] / self._nut_tcp_clear).clamp(0.0, 1.0)
            * finishable
        )

        failed = (info["nut_lost"] | info["nut_spun"]).to(progress.dtype)

        del obs, action
        return reach_reward + 3.0 * progress + preload_reward + withdraw_reward - 2.0 * failed

    def compute_normalized_dense_reward(self, obs, action, info):
        return (self.compute_dense_reward(obs, action, info) + 2.0) / 8.0
