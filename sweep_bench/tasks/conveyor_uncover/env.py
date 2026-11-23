from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.procedural import build_bin
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import tolerance_speed
from sweep_bench.tasks.conveyor_uncover.assets import build_plug_lid
from sweep_bench.tasks.conveyor_uncover.config import RandomizationConfig
from sweep_bench.tasks.conveyor_uncover.scripted import ScriptedUncoverMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorUncover-v0")
@register_env("ConveyorUncover-v0", max_episode_steps=300)
class ConveyorUncoverEnv(ScriptedUncoverMixin, ConveyorWorkspaceEnv):
    """Lift a loose plug lid off an open bin riding the conveyor belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _bin_inner_half: float = 0.045
    _bin_wall_thickness: float = 0.01
    _bin_wall_half_height: float = 0.02
    _bin_floor_half_height: float = 0.006

    _cap_radius: float = 0.058
    _cap_half_thickness: float = 0.004
    _plug_half_height: float = 0.006
    _knob_radius: float = 0.012
    _knob_half_height: float = 0.015

    _bin_patch_radius: float = 0.045

    _conveyor_half_width: float = 0.12

    _finger_gap_tol: float = 0.006
    _lid_to_tcp_threshold: float = 0.03
    _plug_exit_margin: float = 0.02
    _lid_lift_height_above_deck: float = 0.20
    _uncover_sustain_steps: int = 5

    _bin_lift_tol: float = 0.006
    _bin_shove_tol: float = 0.04
    _bin_max_tilt_deg: float = 10.0
    _bin_speed_tol_floor: float = 0.065
    _bin_speed_tol_frac: float = 0.44
    _bin_speed_tol_ceiling: float = 0.15

    default_prompt: str = "take the lid off the bin on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        plug_clearance: float = 0.002,
        **kwargs,
    ) -> None:
        if not 0.0 < plug_clearance < self._bin_inner_half:
            raise ValueError(f"plug_clearance must be in (0, {self._bin_inner_half}) (got {plug_clearance})")
        self.plug_clearance: float = plug_clearance
        self._plug_radius: float = self._bin_inner_half - plug_clearance
        super().__init__(*args, **kwargs)

    @property
    def _bin_outer_half(self) -> float:
        return self._bin_inner_half + 2.0 * self._bin_wall_thickness

    @property
    def _bin_height(self) -> float:
        return 2.0 * self._bin_floor_half_height + 2.0 * self._bin_wall_half_height

    @property
    def _bin_origin_above_deck(self) -> float:
        return self._bin_floor_half_height

    @property
    def _lid_origin_to_cap_bottom(self) -> float:
        return self._knob_half_height + 2.0 * self._cap_half_thickness

    @property
    def _lid_origin_to_plug_bottom(self) -> float:
        return self._lid_origin_to_cap_bottom + 2.0 * self._plug_half_height

    @property
    def _lid_rest_origin_above_deck(self) -> float:
        return self._bin_height + self._lid_origin_to_cap_bottom

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        bin_builder = build_bin(
            self.scene,
            inner_half_x=self._bin_inner_half,
            inner_half_y=self._bin_inner_half,
            wall_thickness=self._bin_wall_thickness,
            wall_half_height=self._bin_wall_half_height,
            floor_half_height=self._bin_floor_half_height,
            color=(0.30, 0.42, 0.62, 1.0),
        )
        bin_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._bin_origin_above_deck])
        self.bin = bin_builder.build(name="uncover_bin")
        self.belt.register_rider(self.bin, patch_radius=self._bin_patch_radius)

        lid_builder = build_plug_lid(
            self.scene,
            knob_radius=self._knob_radius,
            knob_half_height=self._knob_half_height,
            cap_radius=self._cap_radius,
            cap_half_thickness=self._cap_half_thickness,
            plug_radius=self._plug_radius,
            plug_half_height=self._plug_half_height,
        )
        lid_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.lid = lid_builder.build(name="uncover_lid")

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        bin_q = self._sample_bin_orientation(env_idx, b)
        bin_pos = self._sample_bin_position(env_idx, b)
        self.bin.set_pose(Pose.create_from_pq(bin_pos, bin_q))
        bin_vel = torch.zeros((b, 3), device=device)
        bin_vel[:, 1] = self.belt.velocity[env_idx]
        self.bin.set_linear_velocity(bin_vel)
        self.bin.set_angular_velocity(torch.zeros((b, 3), device=device))

        lid_pos = bin_pos.clone()
        lid_pos[:, :2] = lid_pos[:, :2] + self._sample_lid_offset(b)
        lid_pos[:, 2] = deck_z + self._lid_rest_origin_above_deck
        lid_q = torch.zeros((b, 4), device=device)
        lid_q[:, 0] = 1.0
        self.lid.set_pose(Pose.create_from_pq(lid_pos, lid_q))
        self.lid.set_linear_velocity(bin_vel.clone())
        self.lid.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_uncover_trackers(env_idx)

    def _sample_lid_offset(self, b: int) -> torch.Tensor:
        device = self.device
        if not self.randomization.lid_offset:
            return torch.zeros((b, 2), device=device)
        c = self.plug_clearance
        off = self._episode_rng.uniform(-c, c, size=(b, 2)).astype(np.float32)
        return torch.as_tensor(off, device=device)

    def _sample_bin_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.bin_yaw:
            yaw = self._episode_rng.uniform(-np.pi / 4, np.pi / 4, size=b).astype(np.float32)
            yaw_t = torch.as_tensor(yaw, device=device)
            q[:, 0] = torch.cos(yaw_t * 0.5)
            q[:, 3] = torch.sin(yaw_t * 0.5)
        return q

    def _sample_bin_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._bin_outer_half * 1.42 - 0.03
        bin_pos = torch.zeros((b, 3), device=device)
        bin_pos[:, 0] = ox
        bin_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        bin_pos[:, 2] = deck_z + self._bin_origin_above_deck
        return bin_pos

    def _reset_uncover_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_uncover_sustain_counter"):
            self._uncover_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._bin_disturbed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._uncover_sustain_counter[env_idx] = 0
        self._bin_disturbed[env_idx] = False

    def _bin_disturbance(self) -> tuple[torch.Tensor, torch.Tensor]:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        bin_pos = self.bin.pose.p

        supported = torch.abs(bin_pos[:, 1] - oy) < self._conveyor_half_length - self._bin_outer_half

        lifted = torch.abs(bin_pos[:, 2] - (deck_z + self._bin_origin_above_deck)) > self._bin_lift_tol
        shoved = torch.abs(bin_pos[:, 0] - ox) > self._bin_shove_tol
        upright_cos = quaternion_to_matrix(self.bin.pose.q)[:, 2, 2]
        tipped = upright_cos < np.cos(np.deg2rad(self._bin_max_tilt_deg))
        speed_tol = torch.clamp(
            self._bin_speed_tol_frac * tolerance_speed(self.belt.velocity),
            min=self._bin_speed_tol_floor,
            max=self._bin_speed_tol_ceiling,
        )
        slipped = torch.abs(self.bin.linear_velocity[:, 1] - self.belt.velocity) > speed_tol

        return (lifted | shoved | tipped | slipped), supported

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        lid_pos = self.lid.pose.p

        finger_gap = self.gripper_aperture()
        at_knob_width = torch.abs(finger_gap - 2.0 * self._knob_radius) < self._finger_gap_tol
        fingers_on_lid = agent.is_grasping(self.lid)
        lid_at_tcp = torch.linalg.norm(tcp_pos - lid_pos, dim=-1) < self._lid_to_tcp_threshold
        is_held = at_knob_width & fingers_on_lid & lid_at_tcp

        lifted = lid_pos[:, 2] > deck_z + self._lid_lift_height_above_deck

        plug_bottom_z = lid_pos[:, 2] - self._lid_origin_to_plug_bottom
        extracted = plug_bottom_z > deck_z + self._bin_height + self._plug_exit_margin

        disturbed_now, supported = self._bin_disturbance()
        self._bin_disturbed = self._bin_disturbed | (disturbed_now & supported)

        bin_pos = self.bin.pose.p
        bin_on_belt = (
            supported
            & (torch.abs(bin_pos[:, 0] - ox) < self._conveyor_half_width - self._bin_outer_half)
            & (torch.abs(bin_pos[:, 2] - (deck_z + self._bin_origin_above_deck)) < self._bin_lift_tol)
        )

        uncovered = is_held & lifted & bin_on_belt
        self._uncover_sustain_counter = self._advance_sustain(
            "uncover_sustain", self._uncover_sustain_counter, uncovered
        )
        success = (self._uncover_sustain_counter >= self._uncover_sustain_steps) & ~self._bin_disturbed

        return {
            "success": success,
            "is_held": is_held,
            "at_knob_width": at_knob_width,
            "fingers_on_lid": fingers_on_lid,
            "lid_at_tcp": lid_at_tcp,
            "extracted": extracted,
            "lifted": lifted,
            "bin_on_belt": bin_on_belt,
            "bin_disturbed": self._bin_disturbed,
            "bin_supported": supported,
            "plug_above_rim": plug_bottom_z - (deck_z + self._bin_height),
            "uncover_sustain": self._uncover_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        lid_pos = self.lid.pose.p
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        tcp_to_lid = torch.linalg.norm(tcp_pos - lid_pos, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_lid)

        seat_z = deck_z + self._lid_rest_origin_above_deck
        span = self._lid_lift_height_above_deck - self._lid_rest_origin_above_deck
        extract_reward = ((lid_pos[:, 2] - seat_z) / span).clamp(0.0, 1.0)

        reward = reach_reward + 2.0 * extract_reward * info["is_held"].float()
        return reward * (~info["bin_disturbed"]).float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
