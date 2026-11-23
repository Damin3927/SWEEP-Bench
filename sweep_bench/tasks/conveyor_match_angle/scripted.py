from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedMatchAngleMixin(_ScriptedPlaceFamilyMixin):
    carrier: Actor
    _seat_z_carrier: float
    _puck_grip_offset: float
    _cradle_rim_local_z: float
    _scripted_yaw_cmd: torch.Tensor

    _scripted_accel_step: float | None = None

    _scripted_wait_clearance: float = 0.030
    _scripted_place_gap: float = -0.002

    _scripted_center_r_in: float = 0.008
    _scripted_center_r_out: float = 0.050
    _scripted_lower_xy_tol: float = 0.010
    _scripted_release_steps: int = 6

    @property
    def _scripted_rider(self) -> Actor:
        return self.carrier

    def _seat_tcp_z(self, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2] + self._seat_z_carrier + self._puck_grip_offset

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2] + self._cradle_rim_local_z

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seat_z = self._seat_tcp_z(rider_pos)
        return seat_z + self._scripted_wait_clearance, seat_z + self._scripted_place_gap

    def _scripted_release_gripper(self) -> torch.Tensor:
        open_val, _ = self._scripted_gripper_values()
        return torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32)

    def _gripper_down_target_q(self) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

        base_q = super()._gripper_down_target_q()
        if not hasattr(self, "_scripted_yaw_cmd"):
            return base_q
        half = 0.5 * self._scripted_yaw_cmd
        q_yaw = torch.zeros_like(base_q)
        q_yaw[:, 0] = torch.cos(half)
        q_yaw[:, 3] = torch.sin(half)
        return quaternion_multiply(q_yaw, base_q)
