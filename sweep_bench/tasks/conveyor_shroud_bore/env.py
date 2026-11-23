from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply, quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.procedural import build_can
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import tolerance_speed
from sweep_bench.tasks.conveyor_shroud_bore.assets import (
    build_shrouded_port_block,
    validate_port_geometry,
)
from sweep_bench.tasks.conveyor_shroud_bore.config import RandomizationConfig
from sweep_bench.tasks.conveyor_shroud_bore.scripted import ScriptedShroudBoreMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor

_ROD_COLOR = (0.95, 0.62, 0.20, 1.0)


@bench_task("ConveyorShroudBore-v0")
@register_env("ConveyorShroudBore-v0", max_episode_steps=300)
class ConveyorShroudBoreEnv(ScriptedShroudBoreMixin, ConveyorWorkspaceEnv):
    """Insert a held rod into a STEEPLY OBLIQUE bore whose mouth is under a shroud."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _rod_radius: float = 0.006
    _rod_half_length: float = 0.052
    _rod_grasp_offset: float = 0.038

    _bore_length: float = 0.050
    _lead_depth: float = 0.008
    _mouth_radius: float = 0.015
    _wall_thickness: float = 0.005
    _tube_foot: float = 0.014
    _port_segments: int = 12

    _plate_half_extent: float = 0.045
    _base_thickness: float = 0.024
    _port_offset_x: float = 0.012
    _shroud_gap: float = 0.055
    _shroud_overhang: float = 0.016
    _shroud_half_y: float = 0.030
    _shroud_half_thickness: float = 0.004
    _post_half_y: float = 0.020
    _block_density: float = 2500.0

    _block_patch_radius: float = 0.025

    _block_spawn_max_dist: float = 0.40

    _conveyor_half_width: float = 0.10

    _insert_depth: float = 0.030
    _seat_depth: float = 0.034
    _settled_rel_speed_threshold: float = 0.3
    _block_comove_tol_frac: float = 0.25
    _block_comove_tol_floor: float = 0.055
    _block_comove_tol_ceiling: float = 0.100
    _block_rest_z_tol: float = 0.010
    _grip_offset_tol: float = 0.015
    _seat_sustain_steps: int = 5

    default_prompt: str = "insert the rod into the angled port under the shroud on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        port_tilt_deg: float = 40.0,
        bore_radius: float = 0.010,
        **kwargs,
    ) -> None:
        if bore_radius <= self._rod_radius:
            raise ValueError(f"bore_radius={bore_radius} must exceed the rod radius {self._rod_radius}")
        validate_port_geometry(
            bore_radius=bore_radius,
            bore_length=self._bore_length,
            lead_depth=self._lead_depth,
            mouth_radius=self._mouth_radius,
            tilt_rad=math.radians(port_tilt_deg),
            wall_thickness=self._wall_thickness,
            tube_foot=self._tube_foot,
            base_thickness=self._base_thickness,
            port_offset_x=self._port_offset_x,
            plate_half_extent=self._plate_half_extent,
            shroud_gap=self._shroud_gap,
            shroud_overhang=self._shroud_overhang,
            shroud_half_y=self._shroud_half_y,
        )
        self.port_tilt_deg: float = port_tilt_deg
        self._bore_radius: float = bore_radius
        super().__init__(*args, **kwargs)

    @property
    def _port_tilt_rad(self) -> float:
        return math.radians(self.port_tilt_deg)

    @property
    def _tip_below_tcp(self) -> float:
        return self._rod_grasp_offset + self._rod_half_length

    @property
    def keying_alpha_max(self) -> float:
        return self._max_misalignment(self._insert_depth)

    @property
    def keying_max_vertical_engagement(self) -> float:
        tilt = self._port_tilt_rad
        return (2.0 * self._bore_radius - 2.0 * self._rod_radius * math.cos(tilt)) / math.sin(tilt)

    def _max_misalignment(self, engagement: float) -> float:
        r, big_r = self._rod_radius, self._bore_radius
        amp = math.hypot(engagement, 2.0 * r)
        target = 2.0 * big_r / amp
        if target >= 1.0:
            return math.pi / 2.0
        return math.asin(target) - math.atan2(2.0 * r, engagement)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        tilt = self._port_tilt_rad

        block_builder = build_shrouded_port_block(
            self.scene,
            bore_radius=self._bore_radius,
            bore_length=self._bore_length,
            lead_depth=self._lead_depth,
            mouth_radius=self._mouth_radius,
            tilt_rad=tilt,
            wall_thickness=self._wall_thickness,
            tube_foot=self._tube_foot,
            plate_half_extent=self._plate_half_extent,
            base_thickness=self._base_thickness,
            port_offset_x=self._port_offset_x,
            shroud_gap=self._shroud_gap,
            shroud_overhang=self._shroud_overhang,
            shroud_half_y=self._shroud_half_y,
            shroud_half_thickness=self._shroud_half_thickness,
            post_half_y=self._post_half_y,
            density=self._block_density,
            segments=self._port_segments,
        )
        block_builder.initial_pose = sapien.Pose(p=[ox, oy, self.nominal_origin_z])
        self.block = block_builder.build(name="shroud_port_block")
        self.belt.register_rider(self.block, patch_radius=self._block_patch_radius)

        rod_builder = build_can(
            self.scene,
            radius=self._rod_radius,
            half_length=self._rod_half_length,
            color=_ROD_COLOR,
        )
        rod_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.rod = rod_builder.build(name="rod")

        tip = torch.zeros((self.num_envs, 3), device=device)
        tip[:, 2] = -self._rod_half_length
        self._rod_tip_offset = Pose.create_from_pq(p=tip)

        self._port_axis_local = torch.tensor([-math.sin(tilt), 0.0, math.cos(tilt)], dtype=torch.float32, device=device)
        self._throat_offset = Pose.create_from_pq(p=self._port_point_local(self._bore_length).expand(self.num_envs, 3))
        self._mouth_offset = Pose.create_from_pq(
            p=self._port_point_local(self._bore_length + self._lead_depth).expand(self.num_envs, 3)
        )

    def _port_point_local(self, axial: float) -> torch.Tensor:
        tilt = self._port_tilt_rad
        return torch.tensor(
            [[self._port_offset_x - axial * math.sin(tilt), 0.0, axial * math.cos(tilt)]],
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.rod]

    @property
    def nominal_origin_z(self) -> float:
        return self.workspace_offset[2] + self._conveyor_deck_height + self._base_thickness + 0.001

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._rod_radius - self._scripted_hold_margin, dtype=np.float32)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        block_pos = self._sample_block_position(env_idx, b)
        block_q = torch.zeros((b, 4), device=device)
        block_q[:, 0] = 1.0
        self.block.set_pose(Pose.create_from_pq(block_pos, block_q))
        block_vel = torch.zeros((b, 3), device=device)
        block_vel[:, 1] = self.belt.velocity[env_idx]
        self.block.set_linear_velocity(block_vel)
        self.block.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p[env_idx]
        tcp_q = agent.tcp.pose.q[env_idx]
        approach = quaternion_to_matrix(tcp_q)[:, :, 2]
        flip = torch.zeros((b, 4), device=device)
        flip[:, 1] = 1.0
        rod_q = quaternion_multiply(tcp_q, flip)
        self.rod.set_pose(Pose.create_from_pq(tcp_pos + self._rod_grasp_offset * approach, rod_q))
        self.rod.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.rod.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_seat_trackers(env_idx)

    def _sample_block_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, _oz = self.workspace_offset
        device = self.device
        start_y = min(
            self._conveyor_half_length - self._plate_half_extent - 0.04,
            self._block_spawn_max_dist,
        )
        block_pos = torch.zeros((b, 3), device=device)
        block_pos[:, 0] = ox + self._sample_lateral_offset(b)
        block_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        block_pos[:, 2] = self.nominal_origin_z
        return block_pos

    def _sample_lateral_offset(self, b: int) -> torch.Tensor:
        device = self.device
        if not self.randomization.block_lateral:
            return torch.zeros(b, device=device)
        lo, hi = self.randomization.block_lateral_range
        return torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)

    def port_throat_world(self) -> torch.Tensor:
        return (self.block.pose * self._throat_offset).p

    def port_mouth_world(self) -> torch.Tensor:
        return (self.block.pose * self._mouth_offset).p

    def port_axis_world(self) -> torch.Tensor:
        return quaternion_to_matrix(self.block.pose.q) @ self._port_axis_local

    def _rod_tip_world(self) -> torch.Tensor:
        return (self.rod.pose * self._rod_tip_offset).p

    def _tip_axis_coords(self) -> tuple[torch.Tensor, torch.Tensor]:
        axis = self.port_axis_world()
        delta = self._rod_tip_world() - self.port_throat_world()
        axial = (delta * axis).sum(dim=-1)
        radial = torch.linalg.norm(delta - axial.unsqueeze(-1) * axis, dim=-1)
        return -axial, radial

    def _reset_seat_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_seat_sustain_counter"):
            self._seat_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._seat_sustain_counter[env_idx] = 0

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        depth, radial = self._tip_axis_coords()

        clear = self._bore_radius - self._rod_radius
        seated = (radial < clear) & (depth > self._insert_depth) & (depth < self._bore_length)

        tcp_pos = agent.tcp.pose.p
        grip_offset = torch.linalg.norm(self.rod.pose.p - tcp_pos, dim=-1)
        gripped = agent.is_grasping(self.rod) & (
            torch.abs(grip_offset - self._rod_grasp_offset) < self._grip_offset_tol
        )

        rel_speed = torch.linalg.norm(self.rod.linear_velocity - self.block.linear_velocity, dim=-1)
        settled = rel_speed < self._settled_rel_speed_threshold

        block_pos = self.block.pose.p
        running_tol = torch.clamp(
            self._block_comove_tol_frac * tolerance_speed(self.belt.velocity),
            min=self._block_comove_tol_floor,
            max=self._block_comove_tol_ceiling,
        )
        belt_speed_abs = torch.abs(self.belt.velocity)
        comove_tol = torch.where(
            belt_speed_abs > 1e-6,
            torch.minimum(running_tol, 0.5 * tolerance_speed(self.belt.velocity)),
            running_tol,
        )
        rest_z = deck_z + self._base_thickness
        block_on_belt = (
            (torch.abs(block_pos[:, 2] - rest_z) <= self._block_rest_z_tol)
            & (torch.abs(block_pos[:, 0] - ox) < self._conveyor_half_width + self._plate_half_extent)
            & (torch.abs(block_pos[:, 1] - oy) < self._conveyor_half_length + self._plate_half_extent)
            & (torch.abs(self.block.linear_velocity[:, 1] - self.belt.velocity) <= comove_tol)
        )

        holding = seated & gripped & settled & block_on_belt
        self._seat_sustain_counter = self._advance_sustain("seat_sustain", self._seat_sustain_counter, holding)
        success = self._seat_sustain_counter >= self._seat_sustain_steps

        return {
            "success": success,
            "seated": seated,
            "gripped": gripped,
            "settled": settled,
            "block_on_belt": block_on_belt,
            "insert_depth": depth,
            "tip_radial_offset": radial,
            "grip_offset": grip_offset,
            "seat_sustain": self._seat_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        depth, radial = self._tip_axis_coords()

        align_reward = 1 - torch.tanh(20.0 * radial)

        progress = depth.clamp(0.0, self._insert_depth) / self._insert_depth
        aligned = (radial < self._bore_radius - self._rod_radius).float()
        return align_reward + 2.0 * progress * aligned

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
