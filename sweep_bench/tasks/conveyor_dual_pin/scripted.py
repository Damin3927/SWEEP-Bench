from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedDualPinMixin(_ScriptedPlaceFamilyMixin):
    base: Actor
    bracket: Actor
    _base_half_height: float
    _pin_shank_height: float
    _pin_cone_height: float
    _pin_tip_radius: float
    _bracket_half_z: float
    _bracket_pad_half_extent: float
    _bore_chamfer: float
    bore_half_width: float
    _scripted_base_yaw: torch.Tensor

    _scripted_pos_step: float = 0.015
    _scripted_v_ee_max: float = 0.225
    _scripted_accel_step: float | None = 0.0075

    _scripted_v_align: float = 0.15
    _scripted_max_lead: float = 0.30
    _scripted_cross_scale: float = 0.05
    _scripted_prox_scale: float = 0.05

    _scripted_rot_gain: float = 0.6

    _scripted_comove_lead_time: float = 0.04
    _scripted_wait_clearance: float = 0.012
    _scripted_seat_gap: float = 0.004
    _scripted_release_finger_margin: float = 0.022
    _scripted_hold_margin: float = 0.004

    _scripted_withdraw_rim_margin: float = 0.05

    _scripted_center_r_in: float = 0.005
    _scripted_settle_cos: float = 0.996

    _scripted_lower_xy_tol: float = 0.010
    _scripted_lower_z_tol: float = 0.008
    _scripted_release_steps: int = 5

    @property
    def _scripted_rider(self) -> Actor:
        return self.base

    @property
    def _pin_top_above_plate(self) -> float:
        return self._pin_shank_height + self._pin_cone_height

    @property
    def _bore_mouth_capture(self) -> float:
        return self.bore_half_width + self._bore_chamfer - self._pin_tip_radius

    def _scripted_commit_window(self) -> tuple[torch.Tensor, torch.Tensor]:
        agent = cast("Panda", self.agent)
        rot_cos = torch.abs((agent.tcp.pose.q * self._gripper_down_target_q()).sum(dim=-1))
        settled = rot_cos > self._scripted_settle_cos
        r_in = torch.where(
            settled,
            torch.full_like(rot_cos, self._scripted_center_r_in),
            torch.full_like(rot_cos, -1.0),
        )
        r_out = torch.where(
            settled,
            torch.full_like(rot_cos, self._bore_mouth_capture),
            torch.full_like(rot_cos, -0.9),
        )
        return r_in, r_out

    def _plate_top_z(self, deck_z: float) -> float:
        return deck_z + 2.0 * self._base_half_height

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> float:
        return self._plate_top_z(deck_z) + self._pin_top_above_plate

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, float]:
        plate_top = self._plate_top_z(deck_z)
        wait_z = plate_top + self._pin_top_above_plate + self._scripted_wait_clearance + self._bracket_half_z
        lower_z = plate_top + self._scripted_seat_gap + self._bracket_half_z
        return wait_z, lower_z

    def _scripted_gripper_values(self) -> tuple[float, float]:
        open_val, _close_val = super()._scripted_gripper_values()
        hold_target = self._bracket_pad_half_extent - self._scripted_hold_margin
        return open_val, self._gripper_action_for_aperture(2.0 * hold_target)

    def _release_finger_target(self) -> float:
        return min(self._bracket_pad_half_extent + self._scripted_release_finger_margin, 0.04)

    def _scripted_release_gripper(self) -> torch.Tensor:
        value = self._gripper_action_for_aperture(2.0 * self._release_finger_target())
        return torch.full((self.num_envs,), value, device=self.device, dtype=torch.float32)

    def _gripper_down_target_q(self) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

        base_q = super()._gripper_down_target_q()
        if not hasattr(self, "_scripted_base_yaw"):
            return base_q
        half = 0.5 * self._scripted_base_yaw
        q_yaw = torch.zeros_like(base_q)
        q_yaw[:, 0] = torch.cos(half)
        q_yaw[:, 3] = torch.sin(half)
        return quaternion_multiply(q_yaw, base_q)
