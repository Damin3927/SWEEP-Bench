from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedAxleSeatMixin(_ScriptedPlaceFamilyMixin):
    carrier: Actor
    rod: Actor
    _rod_radius: float
    _rod_half_length: float
    _fork_top_local_z: float
    _seated_rod_local_z: float
    fork_pitch: float

    def _rod_axis_samples(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    _scripted_pos_step: float = 0.015
    _scripted_v_ee_max: float = 0.225
    _scripted_accel_step: float | None = 0.00375

    _scripted_v_align: float = 0.15
    _scripted_max_lead: float = 0.30
    _scripted_cross_scale: float = 0.05
    _scripted_prox_scale: float = 0.05

    _scripted_center_r_in: float = 0.010
    _scripted_center_r_out: float = 0.018
    _scripted_lower_xy_tol: float = 0.016

    _scripted_wait_clearance: float = 0.015
    _scripted_seat_press: float = 0.0015

    _scripted_release_steps: int = 6

    _scripted_release_finger_margin: float = 0.005
    _scripted_hold_margin: float = 0.0012

    _scripted_wrist_gate_cos: float = 0.985
    _scripted_capture_margin: float = 0.012

    _scripted_merge_floor_v_lo: float = 0.35
    _scripted_merge_floor_v_hi: float = 0.50

    @property
    def _scripted_rider(self) -> Actor:
        return self.carrier

    def _update_auto_speed_scale(self, tcp_pos: torch.Tensor, rider_pos: torch.Tensor) -> None:
        super()._update_auto_speed_scale(tcp_pos, rider_pos)
        scale = self._scripted_auto_scale
        if scale is None:
            return
        span = self._scripted_merge_floor_v_hi - self._scripted_merge_floor_v_lo
        ramp = ((torch.abs(self.belt.velocity) - self._scripted_merge_floor_v_lo) / span).clamp(0.0, 1.0)
        ceiling = self._scripted_auto_v_ceiling / (self._scripted_pos_step * self.control_freq)
        self._scripted_auto_scale = torch.maximum(scale, 1.0 + (ceiling - 1.0) * ramp)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        if not hasattr(self, "_scripted_wrist_ready"):
            self._scripted_wrist_ready = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self._scripted_dip_ready = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        if env_idx is None:
            self._scripted_wrist_ready[:] = False
            self._scripted_dip_ready[:] = False
        else:
            self._scripted_wrist_ready[env_idx] = False
            self._scripted_dip_ready[env_idx] = False

    def _wrist_settled(self) -> torch.Tensor:
        tcp_q = cast("Panda", self.agent).tcp.pose.q
        rot_cos = torch.abs((tcp_q * self._gripper_down_target_q()).sum(dim=-1))
        self._scripted_wrist_ready = self._scripted_wrist_ready | (rot_cos > self._scripted_wrist_gate_cos)
        return self._scripted_wrist_ready

    def _scripted_notch_offset(self) -> torch.Tensor:
        x, _z, on_rod = self._rod_axis_samples()
        off = x.abs().amax(dim=-1)
        return torch.where(on_rod.all(dim=-1), off, torch.full_like(off, 1e3))

    def _scripted_commit_window(self) -> tuple[torch.Tensor, torch.Tensor]:
        may_dip = self._wrist_settled() & (self._scripted_notch_offset() < self._scripted_capture_margin)
        self._scripted_dip_ready = self._scripted_dip_ready | may_dip
        committed = self._scripted_dip_ready
        slack = self._rod_half_length - self.fork_pitch / 2.0
        zero = torch.zeros_like(committed, dtype=torch.float32)
        r_in = torch.where(
            committed, torch.full_like(zero, slack + self._scripted_center_r_in), torch.full_like(zero, -1e-6)
        )
        r_out = torch.where(committed, torch.full_like(zero, slack + self._scripted_center_r_out), zero)
        return r_in, r_out

    def _scripted_fork_top_z(self, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2] + self._fork_top_local_z

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return self._scripted_fork_top_z(rider_pos)

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wait_z = self._scripted_fork_top_z(rider_pos) + self._scripted_wait_clearance
        lower_z = rider_pos[:, 2] + self._seated_rod_local_z - self._scripted_seat_press
        return wait_z, lower_z

    def _scripted_gripper_values(self) -> tuple[float, float]:
        open_val, _close_val = super()._scripted_gripper_values()
        hold_target = self._rod_radius - self._scripted_hold_margin
        return open_val, self._gripper_action_for_aperture(2.0 * hold_target)

    def _scripted_release_gripper(self) -> torch.Tensor:
        target = self._rod_radius + self._scripted_release_finger_margin
        value = self._gripper_action_for_aperture(2.0 * target)
        return torch.full((self.num_envs,), value, device=self.device, dtype=torch.float32)
