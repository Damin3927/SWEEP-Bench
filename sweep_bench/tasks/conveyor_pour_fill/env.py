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

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import carried_speed
from sweep_bench.tasks.conveyor_pour_fill.assets import (
    build_bead,
    build_bead_jug,
    build_funnel_hopper,
)
from sweep_bench.tasks.conveyor_pour_fill.config import RandomizationConfig
from sweep_bench.tasks.conveyor_pour_fill.scripted import ScriptedPourFillMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor

_JUG_COLOR = (0.86, 0.58, 0.22, 1.0)
_BEAD_COLOR = (0.18, 0.52, 0.86, 1.0)
_HOPPER_COLOR = (0.34, 0.37, 0.43, 1.0)
_FUNNEL_COLOR = (0.90, 0.72, 0.24, 1.0)

_JUG_IN_GRIPPER_Q = (0.0, 1.0, 0.0, 0.0)


@bench_task("ConveyorPourFill-v0")
@register_env("ConveyorPourFill-v0", max_episode_steps=300)
class ConveyorPourFillEnv(ScriptedPourFillMixin, ConveyorWorkspaceEnv):
    """Pour the beads from a held jug into a hopper that is riding away."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _jug_inner_half_x: float = 0.036
    _jug_inner_half_y: float = 0.012
    _jug_inner_height: float = 0.036
    _jug_wall_thickness: float = 0.005
    _jug_grip_wall_thickness: float = 0.008
    _jug_floor_thickness: float = 0.005
    _jug_lid_thickness: float = 0.004
    _jug_spout_lip: float = 0.016
    _jug_spout_height: float = 0.014
    _jug_weir_angle: float = math.radians(40.0)
    _jug_weir_thickness: float = 0.002
    _jug_mount_cant: float = math.radians(32.0)
    _jug_density: float = 500.0
    _jug_grip_margin: float = 0.002

    _bead_count: int = 8
    _bead_radius: float = 0.005
    _bead_density: float = 1200.0
    _bead_angular_damping: float = 5.0
    _bead_pitch_x: float = 0.012
    _bead_pitch_y: float = 0.012

    _hopper_well_half: float = 0.035
    _hopper_well_depth: float = 0.045
    _hopper_wall_thickness: float = 0.006
    _hopper_floor_thickness: float = 0.008
    _hopper_cone_depth: float = 0.028
    _hopper_mouth_half: float = 0.058
    _hopper_density: float = 1900.0
    _hopper_patch_radius: float = 0.030

    _spawn_end_margin: float = 0.12

    _landed_frac_target: float = 0.60
    _bead_comove_tol: float = 0.06
    _well_box_slack: float = 0.003
    _jug_hold_dist: float = 0.055
    _delivery_clear_tol: float = 0.005
    _hopper_upright_cos: float = math.cos(math.radians(12.0))
    _hopper_deck_drop: float = 0.005
    _hopper_on_belt_y: float = 0.85
    _hopper_travel_frac: float = 0.5
    _pour_sustain_steps: int = 10
    _jug_drop_steps: int = 5

    default_prompt: str = "pour the beads from the jug into the moving hopper"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _jug_half_x(self) -> float:
        return self._jug_inner_half_x + self._jug_wall_thickness

    @property
    def _jug_half_y(self) -> float:
        return self._jug_inner_half_y + self._jug_grip_wall_thickness

    @property
    def _jug_half_z(self) -> float:
        return 0.5 * (self._jug_floor_thickness + self._jug_inner_height + self._jug_lid_thickness)

    @property
    def _jug_grasp_width(self) -> float:
        return 2.0 * self._jug_half_y

    @property
    def _jug_weir_run(self) -> float:
        return self._jug_spout_lip / math.tan(self._jug_weir_angle)

    @property
    def _jug_cavity_floor_local_z(self) -> float:
        return -self._jug_half_z + self._jug_floor_thickness

    @property
    def _hopper_floor_local_z(self) -> float:
        return self._hopper_floor_thickness

    @property
    def _hopper_well_rim_local_z(self) -> float:
        return self._hopper_floor_thickness + self._hopper_well_depth

    @property
    def _hopper_mouth_rim_local_z(self) -> float:
        return self._hopper_well_rim_local_z + self._hopper_cone_depth

    @property
    def _hopper_half_footprint(self) -> float:
        return self._hopper_mouth_half

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        hopper_builder = build_funnel_hopper(
            self.scene,
            well_half=self._hopper_well_half,
            well_depth=self._hopper_well_depth,
            wall_thickness=self._hopper_wall_thickness,
            floor_thickness=self._hopper_floor_thickness,
            cone_depth=self._hopper_cone_depth,
            mouth_half=self._hopper_mouth_half,
            body_color=_HOPPER_COLOR,
            funnel_color=_FUNNEL_COLOR,
            density=self._hopper_density,
            initial_pose=sapien.Pose(p=[ox, oy, deck_z + 0.001]),
        )
        self.hopper = hopper_builder.build(name="hopper")
        self.belt.register_rider(self.hopper, patch_radius=self._hopper_patch_radius)

        jug_builder = build_bead_jug(
            self.scene,
            inner_half_x=self._jug_inner_half_x,
            inner_half_y=self._jug_inner_half_y,
            inner_height=self._jug_inner_height,
            wall_thickness=self._jug_wall_thickness,
            grip_wall_thickness=self._jug_grip_wall_thickness,
            floor_thickness=self._jug_floor_thickness,
            lid_thickness=self._jug_lid_thickness,
            spout_lip=self._jug_spout_lip,
            spout_height=self._jug_spout_height,
            weir_angle=self._jug_weir_angle,
            weir_thickness=self._jug_weir_thickness,
            color=_JUG_COLOR,
            density=self._jug_density,
            initial_pose=sapien.Pose(p=[ox, oy, deck_z + 0.30]),
        )
        self.jug = jug_builder.build(name="jug")

        self.beads: list[Actor] = []
        for i, local in enumerate(self._bead_lattice_local()):
            builder = build_bead(
                self.scene,
                radius=self._bead_radius,
                color=_BEAD_COLOR,
                density=self._bead_density,
                initial_pose=sapien.Pose(p=[ox + float(local[0]), oy + float(local[1]), deck_z + 0.40]),
            )
            bead = builder.build(name=f"bead-{i}")
            bead.angular_damping = self._bead_angular_damping
            self.beads.append(bead)

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.jug]

    def _bead_lattice_local(self) -> list[tuple[float, float, float]]:
        cols = self._bead_count // 2
        z = self._jug_cavity_floor_local_z + self._bead_radius + 0.0005
        x0 = -self._jug_inner_half_x + self._jug_weir_run + self._bead_radius + 0.001
        out: list[tuple[float, float, float]] = []
        for j in range(cols):
            x = x0 + j * self._bead_pitch_x
            for sy in (-1.0, 1.0):
                out.append((x, 0.5 * sy * self._bead_pitch_y, z))
        return out

    def _rest_aperture(self, env_idx: torch.Tensor) -> float:
        return self._jug_grasp_width - 2.0 * self._jug_grip_margin

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        oz = self.workspace_offset[2]
        deck_z = oz + self._conveyor_deck_height

        upstream = self._conveyor_half_length - self._hopper_half_footprint - self._spawn_end_margin
        center_xy = self._sample_hopper_center_xy(env_idx, b, upstream)
        hopper_pos = torch.zeros((b, 3), device=device)
        hopper_pos[:, 0] = center_xy[:, 0]
        hopper_pos[:, 1] = center_xy[:, 1]
        hopper_pos[:, 2] = deck_z + 0.001
        self.hopper.set_pose(Pose.create_from_pq(hopper_pos, self._sample_hopper_quat(b)))
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        self.hopper.set_linear_velocity(vel)
        self.hopper.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pose = cast("Panda", self.agent).tcp.pose
        jug_q = self._jug_quat_in_gripper(tcp_pose.q[env_idx])
        jug_pos = tcp_pose.p[env_idx]
        self.jug.set_pose(Pose.create_from_pq(jug_pos, jug_q))
        zero = torch.zeros((b, 3), device=device)
        self.jug.set_linear_velocity(zero)
        self.jug.set_angular_velocity(zero.clone())

        rot = quaternion_to_matrix(jug_q)
        for bead, local in zip(self.beads, self._bead_lattice_local(), strict=True):
            offset = torch.as_tensor(local, dtype=torch.float32, device=device).expand(b, 3)
            world = jug_pos + torch.einsum("bij,bj->bi", rot, offset)
            bead.set_pose(Pose.create_from_pq(world, jug_q.clone()))
            bead.set_linear_velocity(zero.clone())
            bead.set_angular_velocity(zero.clone())

        self._reset_pour_trackers(env_idx)

    def _jug_quat_in_gripper(self, tcp_q: torch.Tensor) -> torch.Tensor:
        b = tcp_q.shape[0]
        half = 0.5 * self._jug_mount_cant
        cant = torch.zeros((b, 4), dtype=torch.float32, device=self.device)
        cant[:, 0] = math.cos(half)
        cant[:, 2] = math.sin(half)
        flip = torch.as_tensor(_JUG_IN_GRIPPER_Q, dtype=torch.float32, device=self.device).expand(b, 4)
        return quaternion_multiply(quaternion_multiply(tcp_q, cant), flip)

    def _sample_hopper_center_xy(self, env_idx: torch.Tensor, b: int, upstream: torch.Tensor | float) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        device = self.device
        xy = torch.zeros((b, 2), device=device)
        xy[:, 0] = ox + self._sample_hopper_dx(b)
        xy[:, 1] = oy - upstream * self.belt.direction_sign[env_idx]
        return xy

    def _sample_hopper_dx(self, b: int) -> torch.Tensor:
        if not self.randomization.hopper_lateral:
            return torch.zeros((b,), device=self.device)
        lo, hi = self.randomization.hopper_x_range
        return torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)

    def _sample_hopper_quat(self, b: int) -> torch.Tensor:
        q = torch.zeros((b, 4), device=self.device)
        if not self.randomization.hopper_yaw:
            q[:, 0] = 1.0
            return q
        lo, hi = self.randomization.hopper_yaw_range
        yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)
        q[:, 0] = torch.cos(0.5 * yaw)
        q[:, 3] = torch.sin(0.5 * yaw)
        return q

    def _reset_pour_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_pour_sustain"):
            zeros_l = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            zeros_b = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._pour_sustain = zeros_l.clone()
            self._jug_loose_steps = zeros_l.clone()
            self._stream_started = zeros_b.clone()
            self._stream_first_exit = zeros_l.clone()
            self._hopper_downed = zeros_b.clone()
            self._jug_dropped = zeros_b.clone()
            beads = (self._bead_count, self.num_envs)
            self._bead_tainted = torch.zeros(beads, dtype=torch.bool, device=self.device)
            self._bead_in_well_prev = torch.zeros(beads, dtype=torch.bool, device=self.device)
            self._bead_in_jug_prev = torch.ones(beads, dtype=torch.bool, device=self.device)
        self._pour_sustain[env_idx] = 0
        self._jug_loose_steps[env_idx] = 0
        self._stream_started[env_idx] = False
        self._stream_first_exit[env_idx] = 0
        self._hopper_downed[env_idx] = False
        self._jug_dropped[env_idx] = False
        self._bead_tainted[:, env_idx] = False
        self._bead_in_well_prev[:, env_idx] = False
        self._bead_in_jug_prev[:, env_idx] = True

    def _bead_positions(self) -> torch.Tensor:
        return torch.stack([bead.pose.p for bead in self.beads], dim=0)

    def _bead_velocities(self) -> torch.Tensor:
        return torch.stack([bead.linear_velocity for bead in self.beads], dim=0)

    def _jug_lowest_z(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.jug.pose.q)
        half = torch.tensor(
            [self._jug_half_x, self._jug_half_y, self._jug_half_z], device=self.device, dtype=torch.float32
        )
        return self.jug.pose.p[:, 2] - (rot[:, 2, :].abs() * half).sum(dim=-1)

    def _beads_in_jug(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.jug.pose.q)
        rel = self._bead_positions() - self.jug.pose.p.unsqueeze(0)
        local = torch.einsum("bji,nbj->nbi", rot, rel)
        s = self._bead_radius
        return (
            (torch.abs(local[..., 0]) < self._jug_inner_half_x + s)
            & (torch.abs(local[..., 1]) < self._jug_inner_half_y + s)
            & (local[..., 2] > self._jug_cavity_floor_local_z - s)
            & (local[..., 2] < self._jug_cavity_floor_local_z + self._jug_inner_height + s)
        )

    def evaluate(self):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        jug_pos = self.jug.pose.p
        hopper_pos = self.hopper.pose.p
        hopper_rot = quaternion_to_matrix(self.hopper.pose.q)
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        bead_pos = self._bead_positions()
        bead_vel = self._bead_velocities()
        rel = bead_pos - hopper_pos.unsqueeze(0)
        local = torch.einsum("bji,nbj->nbi", hopper_rot, rel)
        s = self._well_box_slack
        in_well = (
            (torch.abs(local[..., 0]) < self._hopper_well_half + s)
            & (torch.abs(local[..., 1]) < self._hopper_well_half + s)
            & (local[..., 2] > self._hopper_floor_local_z - s)
            & (local[..., 2] < self._hopper_well_rim_local_z)
        )
        comoving = torch.linalg.norm(bead_vel - self.hopper.linear_velocity.unsqueeze(0), dim=-1) < (
            self._bead_comove_tol
        )
        landed = in_well & comoving

        jug_held = agent.is_grasping(self.jug) & (torch.linalg.norm(tcp_pos - jug_pos, dim=-1) < self._jug_hold_dist)

        mouth_rim_z = hopper_pos[:, 2] + self._hopper_mouth_rim_local_z
        rim_clearance = self._jug_lowest_z() - mouth_rim_z
        jug_clear = rim_clearance >= 0.0

        half_span = self._hopper_half_footprint
        hopper_on_belt = (
            (hopper_pos[:, 2] > deck_z - self._hopper_deck_drop)
            & (torch.abs(hopper_pos[:, 0] - ox) < self._conveyor_half_width + half_span)
            & (torch.abs(hopper_pos[:, 1] - oy) < self._hopper_on_belt_y)
        )
        hopper_up_cos = hopper_rot[:, 2, 2]
        hopper_upright = hopper_up_cos >= self._hopper_upright_cos
        belt_speed = torch.abs(self.belt.velocity)
        along_v = self.hopper.linear_velocity[:, 1] * self.belt.direction_sign
        hopper_travelling = (along_v >= carried_speed(belt_speed, self._hopper_travel_frac)) | (belt_speed <= 1e-6)

        in_jug_now = self._beads_in_jug()
        left_jug = self._bead_in_jug_prev & ~in_jug_now
        arrived = in_well & ~self._bead_in_well_prev
        dirty = (rim_clearance < -self._delivery_clear_tol) | (~hopper_travelling) | (~hopper_upright) | (~jug_held)
        self._bead_tainted = self._bead_tainted | ((arrived | left_jug) & dirty.unsqueeze(0))
        self._bead_in_well_prev = in_well
        self._bead_in_jug_prev = in_jug_now
        landed = landed & ~self._bead_tainted
        landed_frac = landed.float().mean(dim=0)
        enough_landed = landed_frac >= self._landed_frac_target

        advanced = self._eval_advanced("pour_stream")
        in_jug = self._beads_in_jug()
        self._hopper_downed = self._hopper_downed | (~hopper_on_belt)
        just_started = (~in_jug).any(dim=0) & (~self._stream_started)
        self._stream_first_exit = torch.where(just_started, self.elapsed_steps.long(), self._stream_first_exit)
        self._stream_started = self._stream_started | just_started
        loose = self._stream_started & (~jug_held)
        self._jug_loose_steps = torch.where(
            loose, self._jug_loose_steps + advanced.long(), torch.zeros_like(self._jug_loose_steps)
        )
        self._jug_dropped = self._jug_dropped | (self._jug_loose_steps >= self._jug_drop_steps)

        holding = enough_landed & jug_held & jug_clear & hopper_on_belt & hopper_upright & hopper_travelling
        self._pour_sustain = self._advance_sustain("pour_sustain", self._pour_sustain, holding)
        success = (self._pour_sustain >= self._pour_sustain_steps) & (~self._hopper_downed) & (~self._jug_dropped)

        spilled = (bead_pos[..., 2] < deck_z - 0.02).sum(dim=0)
        in_jug_count = in_jug.sum(dim=0)
        settled_count = (in_well & comoving).sum(dim=0)
        in_flight = self._bead_count - in_jug_count - settled_count - spilled
        jug_up_cos = quaternion_to_matrix(self.jug.pose.q)[:, 2, 2]

        return {
            "success": success,
            "landed_frac": landed_frac,
            "enough_landed": enough_landed,
            "jug_held": jug_held,
            "jug_clear": jug_clear,
            "hopper_on_belt": hopper_on_belt,
            "hopper_upright": hopper_upright,
            "hopper_travelling": hopper_travelling,
            "hopper_downed": self._hopper_downed,
            "jug_dropped": self._jug_dropped,
            "pour_sustain": self._pour_sustain,
            "landed_count": landed.sum(dim=0),
            "spilled_count": spilled,
            "in_jug_count": in_jug_count,
            "in_flight_count": in_flight.clamp(min=0),
            "tainted_count": self._bead_tainted.sum(dim=0),
            "stream_started": self._stream_started,
            "stream_first_exit_step": self._stream_first_exit,
            "pour_tilt_deg": torch.rad2deg(torch.arccos(jug_up_cos.clamp(-1.0, 1.0))),
            "jug_to_mouth_xy": torch.linalg.norm(jug_pos[:, :2] - hopper_pos[:, :2], dim=-1),
            "jug_rim_clearance": rim_clearance,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        hopper_pos = self.hopper.pose.p

        station = 1 - torch.tanh(4 * torch.linalg.norm(tcp_pos[:, :2] - hopper_pos[:, :2], dim=-1))
        tip = (info["pour_tilt_deg"] / 90.0).clamp(0.0, 1.0)
        clean = (info["jug_held"] & info["jug_clear"]).float()
        failed = torch.maximum(info["hopper_downed"].float(), info["jug_dropped"].float())
        completion = (
            info["enough_landed"]
            & info["jug_held"]
            & info["jug_clear"]
            & info["hopper_upright"]
            & info["hopper_travelling"]
        ).float()
        return station + clean * (tip + 3.0 * info["landed_frac"]) + completion - 2.0 * failed

    def compute_normalized_dense_reward(self, obs, action, info):
        return (self.compute_dense_reward(obs, action, info) + 2.0) / 8.0
