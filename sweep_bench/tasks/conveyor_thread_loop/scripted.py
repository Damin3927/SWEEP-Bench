from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor
    from mani_skill.utils.structs.pose import Pose


class ScriptedThreadLoopMixin(_ScriptedInsertFamilyMixin):
    carrier: Actor
    rod: Actor
    inner_radius: float
    _rod_radius: float
    _rod_half_length: float
    _rod_grasp_offset: float
    _ring_plane_local_z: float
    _rod_tip_offset: Pose
    _rod_top_offset: Pose
    _passed_down: torch.Tensor
    _exit_clean: torch.Tensor

    _scripted_pos_step: float = 0.015
    _scripted_v_ee_max: float = 0.225
    _scripted_accel_step: float | None = 0.00375

    _scripted_v_align: float = 0.15
    _scripted_max_lead: float = 0.30
    _scripted_cross_scale: float = 0.05
    _scripted_prox_scale: float = 0.05
    _scripted_lead_z_weight: float = 0.0
    _scripted_comove_lead_time: float = 0.12
    _scripted_auto_bias_gain: float = 0.5
    _scripted_hover_z_above: float = 0.32

    _scripted_wait_clearance: float = 0.035
    _scripted_pass_depth: float = 0.070
    _scripted_exit_clearance: float = 0.060

    _scripted_exit_z_rate_frac: float = 0.5

    _scripted_mouth_chamfer: float = 0.012

    _scripted_capture_margin: float = 0.012
    _scripted_dip_latch_depth: float = 0.005

    _scripted_hold_margin: float = 0.0015

    _scripted_rod_aim_gain: float = 0.35
    _scripted_rod_aim_max: float = 0.03

    @property
    def _scripted_rider(self) -> Actor:
        return self.carrier

    @property
    def _scripted_socket_half_width(self) -> float:
        return self.inner_radius - self._rod_radius

    @property
    def _scripted_socket_chamfer(self) -> float:
        return self._scripted_mouth_chamfer

    @property
    def _tip_below_tcp(self) -> float:
        return self._rod_grasp_offset + self._rod_half_length

    def _scripted_ring_plane_z(self, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2] + self._ring_plane_local_z

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        plane_z = self._scripted_ring_plane_z(rider_pos)
        wait_z = plane_z + self._scripted_wait_clearance + self._tip_below_tcp
        lower_z = plane_z - self._scripted_pass_depth + self._tip_below_tcp
        return wait_z, lower_z

    def _rod_plane_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _rod_pierce_offset(self) -> torch.Tensor:
        tip = (self.rod.pose * self._rod_tip_offset).p
        top = (self.rod.pose * self._rod_top_offset).p
        axis = (tip - top) / (2.0 * self._rod_half_length)
        carrier_pos = self.carrier.pose.p
        dz = axis[:, 2]
        dz_safe = torch.where(dz >= 0.0, dz.clamp(min=1e-6), dz.clamp(max=-1e-6))
        t = -(tip[:, 2] - self._scripted_ring_plane_z(carrier_pos)) / dz_safe
        pierce = tip[:, :2] + t.unsqueeze(-1) * axis[:, :2]
        return pierce - carrier_pos[:, :2]

    def _scripted_exit_z_step(self, tip_z_rel: torch.Tensor) -> torch.Tensor:
        frac = torch.where(
            tip_z_rel < 0.0,
            torch.full_like(tip_z_rel, self._scripted_exit_z_rate_frac),
            torch.ones_like(tip_z_rel),
        )
        return self._auto_scaled(self._scripted_pos_step) * frac

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        if not hasattr(self, "_scripted_dip_ready"):
            self._scripted_dip_ready = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        if env_idx is None:
            self._scripted_dip_ready[:] = False
        else:
            self._scripted_dip_ready[env_idx] = False

    def _scripted_commit_window(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_in, base_out = super()._scripted_commit_window()
        tip_z_rel, pierce_r, _on_rod = self._rod_plane_geometry()
        self._scripted_dip_ready = self._scripted_dip_ready | (tip_z_rel <= -self._scripted_dip_latch_depth)
        committed = self._scripted_dip_ready | (pierce_r < self._scripted_capture_margin)
        zero = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        open_in = torch.as_tensor(base_in, dtype=torch.float32, device=self.device).expand(self.num_envs)
        open_out = torch.as_tensor(base_out, dtype=torch.float32, device=self.device).expand(self.num_envs)
        r_in = torch.where(committed, open_in, torch.full_like(zero, -1e-6))
        r_out = torch.where(committed, open_out, zero)
        return r_in, r_out

    def _scripted_hold_gripper(self) -> torch.Tensor:
        target = self._rod_radius - self._scripted_hold_margin
        value = self._gripper_action_for_aperture(2.0 * target)
        return torch.full((self.num_envs, 1), value, device=self.device, dtype=torch.float32)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_scripted_motion_dir"):
            self.reset_scripted()

        state = self._intercept_state()
        rider_pos = self._scripted_rider.pose.p
        tip_z_rel, _pierce_r, _on_rod = self._rod_plane_geometry()

        in_exit = self._passed_down
        in_hold = self._exit_clean

        lead = (state.speed * self._scripted_comove_lead_time + self._scripted_auto_lead_bias).unsqueeze(-1)
        exit_xy = rider_pos[:, :2] + state.direction * lead

        z_rate = torch.as_tensor(
            self._scripted_exit_z_step(tip_z_rel),
            dtype=state.tcp_pos.dtype,
            device=state.tcp_pos.device,
        ).expand(self.num_envs)
        exit_z_final = self._scripted_ring_plane_z(rider_pos) + self._scripted_exit_clearance + self._tip_below_tcp
        exit_z = torch.minimum(state.tcp_pos[:, 2] + z_rate, exit_z_final)

        target_world = state.target_world.clone()
        aim_fix = (self._rod_pierce_offset() * self._scripted_rod_aim_gain).clamp(
            -self._scripted_rod_aim_max, self._scripted_rod_aim_max
        )
        target_world[:, 0] = target_world[:, 0] - aim_fix[:, 0]
        target_world[:, 1] = target_world[:, 1] - aim_fix[:, 1]
        target_world[:, 0] = torch.where(in_exit, exit_xy[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(in_exit, exit_xy[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(in_exit, exit_z, target_world[:, 2])
        target_world[:, 0] = torch.where(in_hold, state.tcp_pos[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(in_hold, state.tcp_pos[:, 1], target_world[:, 1])

        pos_input = self._position_input(target_world, state.tcp_pos)
        rot_input = self._rotation_input(state.tcp_q)
        gripper = self._scripted_hold_gripper()

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
