from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import carried_speed, tolerance_speed
from sweep_bench.tasks.conveyor_side_label.assets import build_label_pad, build_marked_box
from sweep_bench.tasks.conveyor_side_label.config import RandomizationConfig
from sweep_bench.tasks.conveyor_side_label.scripted import ScriptedSideLabelMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorSideLabel-v0")
@register_env("ConveyorSideLabel-v0", max_episode_steps=300)
class ConveyorSideLabelEnv(ScriptedSideLabelMixin, ConveyorWorkspaceEnv):
    """Press a held label pad flat onto the printed patch on a moving box's side face."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    conveyor_speed: float = 0.10

    _box_half_x: float = 0.045
    _box_half_y: float = 0.035
    _box_half_z: float = 0.025

    _box_density: float = 2600.0

    _box_patch_radius: float = 0.030

    _patch_half_y: float = 0.018
    _patch_half_z: float = 0.009
    _patch_default_y: float = 0.0
    _patch_default_z: float = 0.0

    _pad_half_x: float = 0.006
    _pad_half_y: float = 0.024
    _pad_half_z: float = 0.012
    _pad_density: float = 900.0
    _pad_grasp_offset: float = 0.010
    _pad_z_offset: float = 0.006

    _contact_gap: float = 0.003
    _overlap_frac: float = 0.7
    _press_cone_deg: float = 20.0
    _held_distance: float = 0.05
    _band_half: float = 0.08
    _deck_rest_tol: float = 0.020
    _tip_max_deg: float = 15.0
    _ride_v_tol_floor: float = 0.06
    _ride_v_tol_speed_frac: float = 0.20
    _ride_v_min_frac: float = 0.5
    _ride_v_stall_floor: float = 0.01
    _drop_below_deck: float = 0.02
    _label_sustain_steps: int = 12

    default_prompt: str = "press the label pad onto the printed patch on the side of the box on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        randomization: RandomizationConfig | None = None,
        reconfiguration_freq: int | None = None,
        num_envs: int = 1,
        **kwargs,
    ) -> None:
        randomization = randomization if randomization is not None else self.RandomizationConfig()
        if reconfiguration_freq is None:
            load_time_axes = randomization.patch_offset or randomization.patch_height
            reconfiguration_freq = 1 if (num_envs == 1 and load_time_axes) else 0
        super().__init__(
            *args, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, randomization=randomization, **kwargs
        )

    @property
    def _face_ahead_of_tcp(self) -> float:
        return self._pad_grasp_offset + self._pad_half_x

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        b = self.num_envs

        rcfg = self.randomization
        rng = self._batched_episode_rng
        if rcfg.patch_offset:
            lo, hi = rcfg.patch_offset_range
            patch_y = np.asarray(rng.uniform(lo, hi), dtype=np.float32)
        else:
            patch_y = np.full(b, self._patch_default_y, dtype=np.float32)
        if rcfg.patch_height:
            lo, hi = rcfg.patch_height_range
            patch_z = np.asarray(rng.uniform(lo, hi), dtype=np.float32)
        else:
            patch_z = np.full(b, self._patch_default_z, dtype=np.float32)

        patch_local = torch.zeros((b, 3), device=device)
        patch_local[:, 0] = -self._box_half_x
        patch_local[:, 1] = torch.as_tensor(patch_y, device=device)
        patch_local[:, 2] = torch.as_tensor(patch_z, device=device)
        self._patch_pose = Pose.create_from_pq(p=patch_local)

        boxes: list[Actor] = []
        for i in range(b):
            builder = build_marked_box(
                self.scene,
                half_x=self._box_half_x,
                half_y=self._box_half_y,
                half_z=self._box_half_z,
                density=self._box_density,
                patch_y=float(patch_y[i]),
                patch_z=float(patch_z[i]),
                patch_half_y=self._patch_half_y,
                patch_half_z=self._patch_half_z,
            )
            builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._box_half_z + 0.001])
            builder.set_scene_idxs([i])
            box = builder.build(name=f"marked_box-{i}")
            self.remove_from_state_dict_registry(box)
            boxes.append(box)
        self.box = Actor.merge(boxes, name="marked_box")
        self.add_to_state_dict_registry(self.box)
        self.belt.register_rider(self.box, patch_radius=self._box_patch_radius)

        pad_builder = build_label_pad(
            self.scene,
            half_x=self._pad_half_x,
            half_y=self._pad_half_y,
            half_z=self._pad_half_z,
            density=self._pad_density,
        )
        pad_builder.initial_pose = sapien.Pose(p=[ox - 0.2, oy, deck_z + 0.5])
        self.pad = pad_builder.build(name="label_pad")

        face = torch.zeros((b, 3), device=device)
        face[:, 0] = self._pad_half_x
        self._pad_face_pose = Pose.create_from_pq(p=face)

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._pad_half_y - self._scripted_hold_margin, dtype=np.float32)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        box_pos = self._sample_box_position(env_idx, b)
        box_q = torch.zeros((b, 4), device=device)
        box_q[:, 0] = 1.0
        self.box.set_pose(Pose.create_from_pq(box_pos, box_q))
        box_vel = torch.zeros((b, 3), device=device)
        box_vel[:, 1] = self.belt.velocity[env_idx]
        self.box.set_linear_velocity(box_vel)
        self.box.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pose = cast("Panda", self.agent).tcp.pose[env_idx]
        offset = torch.zeros((b, 3), device=device)
        offset[:, 0] = self._pad_grasp_offset
        offset[:, 2] = -self._pad_z_offset
        self.pad.set_pose(tcp_pose * Pose.create_from_pq(p=offset))
        self.pad.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.pad.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_label_trackers(env_idx, box_pos, box_q)

    def _sample_box_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._box_half_y - 0.04
        box_pos = torch.zeros((b, 3), device=device)
        box_pos[:, 0] = ox
        if self.randomization.box_lateral:
            lo, hi = self.randomization.box_lateral_range
            offsets = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            box_pos[:, 0] = ox + torch.as_tensor(offsets, device=device)
        box_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        box_pos[:, 2] = deck_z + self._box_half_z + 0.001
        return box_pos

    def _reset_label_trackers(self, env_idx: torch.Tensor, box_pos: torch.Tensor, box_q: torch.Tensor) -> None:
        from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

        if not hasattr(self, "_label_sustain_counter"):
            self._label_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._drop_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._box_spawn_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._box_spawn_up = torch.zeros((self.num_envs, 3), device=self.device)
            self._box_spawn_up[:, 2] = 1.0
        self._label_sustain_counter[env_idx] = 0
        self._drop_latch[env_idx] = False
        self._box_spawn_pos[env_idx] = box_pos
        self._box_spawn_up[env_idx] = quaternion_to_matrix(box_q)[:, :, 2]

    def evaluate(self):
        from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

        box_pose = self.box.pose
        pad_pose = self.pad.pose
        r_box = quaternion_to_matrix(box_pose.q)
        r_pad = quaternion_to_matrix(pad_pose.q)

        face_local = (box_pose.inv() * (pad_pose * self._pad_face_pose)).p
        patch_local = self._patch_pose.p

        face_gap = -self._box_half_x - face_local[:, 0]
        contact = face_gap.abs() < self._contact_gap

        ov_y = (
            torch.minimum(face_local[:, 1] + self._pad_half_y, patch_local[:, 1] + self._patch_half_y)
            - torch.maximum(face_local[:, 1] - self._pad_half_y, patch_local[:, 1] - self._patch_half_y)
        ).clamp(min=0.0)
        ov_z = (
            torch.minimum(face_local[:, 2] + self._pad_half_z, patch_local[:, 2] + self._patch_half_z)
            - torch.maximum(face_local[:, 2] - self._pad_half_z, patch_local[:, 2] - self._patch_half_z)
        ).clamp(min=0.0)
        overlap_frac = (ov_y * ov_z) / (4.0 * self._patch_half_y * self._patch_half_z)
        coverage = overlap_frac >= self._overlap_frac

        cos_cone = math.cos(math.radians(self._press_cone_deg))
        m_rel = torch.einsum("bki,bkj->bij", r_box, r_pad)
        press_cos = m_rel[:, 0, 0].clamp(-1.0, 1.0)
        roll_cos = m_rel[:, 1, 1].abs().clamp(max=1.0)
        flat = (press_cos > cos_cone) & (roll_cos > cos_cone)

        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        held = torch.linalg.norm(tcp_pos - pad_pose.p, dim=-1) < self._held_distance

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        box_pos = box_pose.p
        extent_x = (
            r_box[:, 0, 0].abs() * self._box_half_x
            + r_box[:, 0, 1].abs() * self._box_half_y
            + r_box[:, 0, 2].abs() * self._box_half_z
        )
        box_shift_x = (box_pos[:, 0] - self._box_spawn_pos[:, 0]).abs()
        on_band = (box_pos[:, 0] - ox).abs() + extent_x < self._band_half
        on_deck = (box_pos[:, 2] - self._box_spawn_pos[:, 2]).abs() < self._deck_rest_tol

        up_now = r_box[:, :, 2]
        tilt_cos = (up_now * self._box_spawn_up).sum(dim=-1).clamp(-1.0, 1.0)
        untipped = tilt_cos > math.cos(math.radians(self._tip_max_deg))

        belt_speed = self.belt.velocity.abs()
        v_y = self.box.linear_velocity[:, 1]
        v_tolerance = self._ride_v_tol_floor + self._ride_v_tol_speed_frac * tolerance_speed(self.belt.velocity)
        v_slip = (v_y - self.belt.velocity).abs()
        v_carry = v_y * self.belt.direction_sign
        carried = v_carry >= carried_speed(belt_speed, self._ride_v_min_frac) - self._ride_v_stall_floor
        riding = (v_slip < v_tolerance) & carried

        self._drop_latch = self._drop_latch | (box_pos[:, 2] < deck_z - self._drop_below_deck)
        not_dropped = ~self._drop_latch

        holding = contact & coverage & flat & held & on_band & on_deck & untipped & riding & not_dropped
        self._label_sustain_counter = self._advance_sustain("label_sustain", self._label_sustain_counter, holding)
        success = self._label_sustain_counter >= self._label_sustain_steps

        return {
            "success": success,
            "contact": contact,
            "coverage": coverage,
            "flat": flat,
            "held": held,
            "on_band": on_band,
            "on_deck": on_deck,
            "untipped": untipped,
            "riding": riding,
            "not_dropped": not_dropped,
            "face_gap": face_gap,
            "overlap_frac": overlap_frac,
            "press_deg": torch.rad2deg(torch.acos(press_cos)),
            "pad_roll_deg": torch.rad2deg(torch.acos(roll_cos)),
            "box_shift_x": box_shift_x,
            "box_tilt_deg": torch.rad2deg(torch.acos(tilt_cos)),
            "box_v_slip": v_slip,
            "box_v_carry": v_carry,
            "belt_y_extent": (box_pos[:, 1] - oy).abs(),
        }

    def compute_dense_reward(self, obs, action, info):
        box_pose = self.box.pose
        face_local = (box_pose.inv() * (self.pad.pose * self._pad_face_pose)).p
        patch_local = self._patch_pose.p

        err = torch.sqrt(
            ((face_local[:, 1] - patch_local[:, 1]) / self._patch_half_y) ** 2
            + ((face_local[:, 2] - patch_local[:, 2]) / self._patch_half_z) ** 2
            + 1e-9
        )
        align_reward = 1 - torch.tanh(2.0 * err)

        approach = 1.0 - (info["face_gap"].clamp(0.0, 0.05) / 0.05)
        return align_reward + 2.0 * approach * info["coverage"].float() + 2.0 * (info["contact"] & info["flat"]).float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
