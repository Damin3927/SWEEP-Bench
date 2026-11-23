from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedHookRailMixin(_ScriptedPlaceFamilyMixin):
    rack: Actor
    _bar_above_rack_origin: float
    _bar_radius: float
    _seat_local_z: float
    _throat_depth: float
    _saddle_bottom_local_z: float
    _saddle_grasp_half: float
    _finger_face_offset: float

    _scripted_grip_squeeze: float = 0.002

    _scripted_accel_step: float | None = 0.006

    _scripted_comove_lead_time: float = 0.0

    _scripted_wait_clearance: float = 0.013
    _scripted_seat_overshoot: float = 0.002
    _scripted_release_finger_margin: float = 0.008

    _scripted_center_r_in: float = 0.040
    _scripted_center_r_out: float = 0.070
    _scripted_lower_xy_tol: float = 0.010
    _scripted_lower_z_tol: float = 0.006
    _scripted_release_steps: int = 4
    _scripted_withdraw_rim_margin: float = 0.0

    _scripted_auto_dip_slip_lo: float = 1.0e3
    _scripted_auto_dip_slip_hi: float = 2.0e3
    _scripted_withdraw_z_above: float = 0.24

    _scripted_reach_y_max: float = 0.40
    _scripted_commit_rot_cos: float = 0.990

    @property
    def _scripted_rider(self) -> Actor:
        return self.rack

    def _position_input(self, target_world: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
        oy = self.workspace_offset[1]
        clamped = target_world.clone()
        clamped[:, 1] = clamped[:, 1].clamp(oy - self._scripted_reach_y_max, oy + self._scripted_reach_y_max)
        return super()._position_input(clamped, tcp_pos)

    def _scripted_commit_window(self) -> tuple[torch.Tensor, torch.Tensor]:
        tcp_q = cast("Panda", self.agent).tcp.pose.q
        rot_cos = torch.abs((tcp_q * self._gripper_down_target_q()).sum(dim=-1))
        settled = rot_cos > self._scripted_commit_rot_cos
        r_in = torch.where(settled, torch.full_like(rot_cos, self._scripted_center_r_in), -torch.ones_like(rot_cos))
        r_out = torch.where(settled, torch.full_like(rot_cos, self._scripted_center_r_out), torch.zeros_like(rot_cos))
        return r_in, r_out

    def _bar_world_z(self, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2] + self._bar_above_rack_origin

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return self._bar_world_z(rider_pos) + self._throat_depth

    def _scripted_wait_lower_z(
        self, deck_z: float, rider_pos: torch.Tensor
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        bar_z = self._bar_world_z(rider_pos)
        wait_z = bar_z + self._bar_radius - self._saddle_bottom_local_z + self._scripted_wait_clearance
        lower_z = bar_z - self._seat_local_z - self._scripted_seat_overshoot
        return wait_z, lower_z

    def _finger_target(self, face_half_width: float) -> float:
        return min(max(face_half_width - self._finger_face_offset, -0.01), 0.04)

    def _scripted_gripper_values(self) -> tuple[float, float]:
        open_val, _close_val = super()._scripted_gripper_values()
        target = self._finger_target(self._saddle_grasp_half - self._scripted_grip_squeeze)
        return open_val, self._gripper_action_for_aperture(2.0 * target)

    def _release_finger_target(self) -> float:
        return self._finger_target(self._saddle_grasp_half + self._scripted_release_finger_margin)

    def _scripted_release_gripper(self) -> torch.Tensor:
        value = self._gripper_action_for_aperture(2.0 * self._release_finger_target())
        return torch.full((self.num_envs,), value, device=self.device, dtype=torch.float32)
