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
from sweep_bench.tasks.conveyor_base.on_belt import carried_speed, tolerance_speed
from sweep_bench.tasks.conveyor_scan_aim.assets import build_code_block, build_scanner_wand
from sweep_bench.tasks.conveyor_scan_aim.config import RandomizationConfig
from sweep_bench.tasks.conveyor_scan_aim.scripted import ScriptedScanAimMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_WAND_IN_HAND_Q = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)


@bench_task("ConveyorScanAim-v0")
@register_env("ConveyorScanAim-v0", max_episode_steps=300)
class ConveyorScanAimEnv(ScriptedScanAimMixin, ConveyorWorkspaceEnv):
    """Hold a held scanner wand aimed at a moving code patch, without touching it."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _handle_half_width: float = 0.007
    _handle_half_length: float = 0.030
    _muzzle_radius: float = 0.0045
    _muzzle_half_length: float = 0.020
    _wand_grasp_offset: float = 0.018

    _block_half_x: float = 0.040
    _block_half_y: float = 0.040
    _block_half_z: float = 0.045
    _block_patch_radius: float = 0.030

    _patch_local_z_nominal: float = 0.015
    _patch_inset: float = 0.008

    _standoff_lo: float = 0.045
    _standoff_hi: float = 0.075
    _standoff_nominal: float = 0.060

    _aim_cone_deg: float = 10.0
    _finger_gap_tol: float = 0.003
    _contact_force_threshold: float = 0.5
    _ride_x_tolerance: float = 0.023
    _ride_z_tolerance: float = 0.006
    _ride_v_tolerance_floor: float = 0.06
    _ride_v_tolerance_speed_frac: float = 0.20
    _ride_v_min_frac: float = 0.5
    _ride_v_stall_floor: float = 0.01
    _ride_turn_tolerance: float = 8.0
    _ride_y_limit: float = 0.85
    _aim_sustain_steps: int = 12

    default_prompt: str = "hold the scanner aimed at the code patch on the conveyor belt without touching it"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        patch_half: float = 0.020,
        randomization: RandomizationConfig | None = None,
        reconfiguration_freq: int | None = None,
        num_envs: int = 1,
        **kwargs,
    ) -> None:
        if patch_half <= self._patch_inset:
            raise ValueError(f"patch_half={patch_half} leaves no lateral budget after the {self._patch_inset} m inset")
        self.patch_half: float = patch_half
        randomization = randomization if randomization is not None else self.RandomizationConfig()
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if (num_envs == 1 and randomization.patch_height) else 0
        self._patch_local_z: float = self._patch_local_z_nominal
        super().__init__(
            *args, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, randomization=randomization, **kwargs
        )

    @property
    def _muzzle_centre_local_z(self) -> float:
        return self._handle_half_length + self._muzzle_half_length

    @property
    def _muzzle_ahead_of_tcp(self) -> float:
        return self._wand_grasp_offset + self._muzzle_centre_local_z

    @property
    def _block_circumradius(self) -> float:
        return math.hypot(self._block_half_x, self._block_half_y)

    def _patch_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rot = quaternion_to_matrix(self.block.pose.q)
        normal = -rot[:, :, 0]
        u = rot[:, :, 1]
        w = rot[:, :, 2]
        centre = self.block.pose.p + normal * self._block_half_x + w * self._patch_local_z
        return centre, normal, u, w

    def _muzzle(self) -> tuple[torch.Tensor, torch.Tensor]:
        rot = quaternion_to_matrix(self.wand.pose.q)
        axis = rot[:, :, 2]
        centre = self.wand.pose.p + axis * self._muzzle_centre_local_z
        return centre, axis

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        self._patch_local_z = self._patch_local_z_nominal
        if self.randomization.patch_height:
            lo, hi = self.randomization.patch_height_range
            self._patch_local_z = self._patch_local_z_nominal + float(self._episode_rng.uniform(lo, hi))

        block_builder = build_code_block(
            self.scene,
            half_x=self._block_half_x,
            half_y=self._block_half_y,
            half_z=self._block_half_z,
            patch_half=self.patch_half,
            patch_local_z=self._patch_local_z,
        )
        block_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._block_half_z + 0.001])
        self.block = block_builder.build(name="code_block")
        self.belt.register_rider(self.block, patch_radius=self._block_patch_radius)

        wand_builder = build_scanner_wand(
            self.scene,
            handle_half_width=self._handle_half_width,
            handle_half_length=self._handle_half_length,
            muzzle_radius=self._muzzle_radius,
            muzzle_half_length=self._muzzle_half_length,
        )
        wand_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.wand = wand_builder.build(name="scanner_wand")

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._handle_half_width - self._scripted_hold_margin, dtype=np.float32)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        block_pos = self._sample_block_position(env_idx, b)
        lo, hi = self.randomization.patch_yaw_range
        yaw = torch.as_tensor(np.deg2rad(self._episode_rng.uniform(lo, hi, size=b)).astype(np.float32), device=device)
        block_q = torch.zeros((b, 4), device=device)
        block_q[:, 0] = torch.cos(0.5 * yaw)
        block_q[:, 3] = torch.sin(0.5 * yaw)
        self.block.set_pose(Pose.create_from_pq(block_pos, block_q))
        block_vel = torch.zeros((b, 3), device=device)
        block_vel[:, 1] = self.belt.velocity[env_idx]
        self.block.set_linear_velocity(block_vel)
        self.block.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pose = cast("Panda", self.agent).tcp.pose
        tcp_pos = tcp_pose.p[env_idx]
        tcp_q = tcp_pose.q[env_idx]
        in_hand = torch.as_tensor(_WAND_IN_HAND_Q, dtype=tcp_q.dtype, device=device).unsqueeze(0).repeat(b, 1)
        wand_q = quaternion_multiply(tcp_q, in_hand)
        body_x = quaternion_to_matrix(tcp_q)[:, :, 0]
        wand_pos = tcp_pos + body_x * self._wand_grasp_offset
        self.wand.set_pose(Pose.create_from_pq(wand_pos, wand_q))
        self.wand.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.wand.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_aim_trackers(env_idx, block_pos, block_q)

        target = np.full(b, self._standoff_nominal, dtype=np.float32)
        if self.randomization.standoff_target:
            slo, shi = self.randomization.standoff_target_range
            target = self._episode_rng.uniform(slo, shi, size=b).astype(np.float32)
        self._standoff_target[env_idx] = torch.as_tensor(target, device=device)

    def _sample_block_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._block_circumradius - 0.04
        block_pos = torch.zeros((b, 3), device=device)
        block_pos[:, 0] = ox
        if self.randomization.block_lateral:
            lo, hi = self.randomization.block_lateral_range
            offsets = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            block_pos[:, 0] = ox + torch.as_tensor(offsets, device=device)
        block_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        block_pos[:, 2] = deck_z + self._block_half_z + 0.001
        return block_pos

    def _reset_aim_trackers(self, env_idx: torch.Tensor, block_pos: torch.Tensor, block_q: torch.Tensor) -> None:
        if not hasattr(self, "_aim_sustain_counter"):
            self._aim_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._wand_touched_block = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._block_spawn_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._block_spawn_q = torch.zeros((self.num_envs, 4), device=self.device)
            self._block_spawn_q[:, 0] = 1.0
            self._standoff_target = torch.full((self.num_envs,), self._standoff_nominal, device=self.device)
        self._aim_sustain_counter[env_idx] = 0
        self._wand_touched_block[env_idx] = False
        self._block_spawn_pos[env_idx] = block_pos
        self._block_spawn_q[env_idx] = block_q

    def _update_contact_latch(self) -> None:
        agent = cast("Panda", self.agent)
        for other in (self.wand, agent.finger1_link, agent.finger2_link):
            force = torch.linalg.norm(self.scene.get_pairwise_contact_forces(other, self.block), dim=-1)
            self._wand_touched_block = self._wand_touched_block | (force > self._contact_force_threshold)

    def evaluate(self):
        patch_centre, normal, u, w = self._patch_frame()
        muzzle_centre, muzzle_axis = self._muzzle()

        rel = muzzle_centre - patch_centre
        standoff = (rel * normal).sum(dim=-1)
        in_band = (standoff > self._standoff_lo) & (standoff < self._standoff_hi)

        aim_cos = -(muzzle_axis * normal).sum(dim=-1)
        aim_deg = torch.rad2deg(torch.acos(aim_cos.clamp(-1.0, 1.0)))
        aimed = aim_deg < self._aim_cone_deg

        denom = (muzzle_axis * normal).sum(dim=-1).clamp(max=-1e-3)
        hit = rel - (standoff / denom).unsqueeze(-1) * muzzle_axis
        hit_u = (hit * u).sum(dim=-1)
        hit_w = (hit * w).sum(dim=-1)
        lateral_limit = self.patch_half - self._patch_inset
        on_patch = (hit_u.abs() < lateral_limit) & (hit_w.abs() < lateral_limit)

        finger_gap = self.gripper_aperture()
        gripped = torch.abs(finger_gap - 2.0 * self._handle_half_width) < self._finger_gap_tol

        self._update_contact_latch()
        no_contact = ~self._wand_touched_block

        ox, oy, _oz = self.workspace_offset
        block_pos = self.block.pose.p
        belt_speed = self.belt.velocity.abs()
        v_y = self.block.linear_velocity[:, 1]
        tol_speed = tolerance_speed(self.belt.velocity)
        v_tolerance = self._ride_v_tolerance_floor + self._ride_v_tolerance_speed_frac * tol_speed
        v_slip = (v_y - self.belt.velocity).abs()
        min_carry = carried_speed(belt_speed, self._ride_v_min_frac) - self._ride_v_stall_floor
        carried = v_y * self.belt.direction_sign >= min_carry
        q_dot = (self.block.pose.q * self._block_spawn_q).sum(dim=-1).abs().clamp(max=1.0)
        turn_deg = torch.rad2deg(2.0 * torch.acos(q_dot))
        riding = (
            ((block_pos[:, 0] - ox).abs() < self._ride_x_tolerance)
            & ((block_pos[:, 2] - self._block_spawn_pos[:, 2]).abs() < self._ride_z_tolerance)
            & (v_slip < v_tolerance)
            & carried
            & (turn_deg < self._ride_turn_tolerance)
            & ((block_pos[:, 1] - oy).abs() <= self._ride_y_limit)
        )

        holding = in_band & aimed & on_patch & gripped & no_contact & riding
        self._aim_sustain_counter = self._advance_sustain("aim_sustain", self._aim_sustain_counter, holding)
        success = self._aim_sustain_counter >= self._aim_sustain_steps

        return {
            "success": success,
            "in_band": in_band,
            "aimed": aimed,
            "on_patch": on_patch,
            "gripped": gripped,
            "no_contact": no_contact,
            "riding": riding,
            "standoff": standoff,
            "aim_deg": aim_deg,
            "hit_u": hit_u,
            "hit_w": hit_w,
            "finger_gap": finger_gap,
            "block_turn_deg": turn_deg,
            "block_v_slip": v_slip,
            "aim_sustain": self._aim_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        band_centre = 0.5 * (self._standoff_lo + self._standoff_hi)
        band_half = 0.5 * (self._standoff_hi - self._standoff_lo)
        standoff_reward = 1.0 - torch.tanh(2.0 * (info["standoff"] - band_centre).abs() / band_half)
        aim_reward = 1.0 - torch.tanh(info["aim_deg"] / self._aim_cone_deg)
        lateral = torch.sqrt(info["hit_u"] ** 2 + info["hit_w"] ** 2 + 1e-12)
        lateral_reward = 1.0 - torch.tanh(2.0 * lateral / (self.patch_half - self._patch_inset))
        held = (info["in_band"] & info["aimed"] & info["on_patch"] & info["gripped"] & info["riding"]).float()
        return (standoff_reward + aim_reward + lateral_reward + 2.0 * held) * info["no_contact"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
