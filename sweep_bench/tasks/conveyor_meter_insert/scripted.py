from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)
from sweep_bench.tasks.conveyor_base.scripted_families import (
    _smoothstep,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedMeterInsertMixin(_ScriptedInsertFamilyMixin):
    block: Actor
    rod: Actor
    _block_half_height: float
    clear_radius: float
    _bore_half_width_nominal: float
    _chamfer: float
    _rod_radius: float
    _rod_grasp_offset: float
    _rod_half_length: float
    _target_depth: torch.Tensor

    _scripted_pos_step: float = 0.015
    _scripted_v_ee_max: float = 0.225
    _scripted_accel_step: float | None = 0.00375

    _scripted_v_align: float = 0.15
    _scripted_max_lead: float = 0.30
    _scripted_cross_scale: float = 0.05
    _scripted_prox_scale: float = 0.05

    _lead_time_lo: float = 0.08
    _lead_time_hi: float = 0.12
    _lead_v_lo: float = 0.30
    _lead_v_hi: float = 0.50

    @property
    def _scripted_comove_lead_time(self) -> torch.Tensor:
        speed = torch.abs(self.belt.velocity)
        ramp = _smoothstep((speed - self._lead_v_lo) / (self._lead_v_hi - self._lead_v_lo))
        return self._lead_time_lo + (self._lead_time_hi - self._lead_time_lo) * ramp

    _scripted_wait_clearance: float = 0.012

    _scripted_depth_bias: float = -0.003

    _scripted_hold_margin: float = 0.0025

    @property
    def _scripted_rider(self) -> Actor:
        return self.block

    @property
    def _scripted_socket_half_width(self) -> float:
        return self.clear_radius

    @property
    def _scripted_socket_chamfer(self) -> float:
        return self._chamfer

    @property
    def _tip_below_tcp(self) -> float:
        return self._rod_grasp_offset + self._rod_half_length

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mouth_z = rider_pos[:, 2] + self._block_half_height
        tip_below = self._tip_below_tcp
        wait_z = mouth_z + self._chamfer + self._scripted_wait_clearance + tip_below
        lower_z = mouth_z - (self._target_depth + self._scripted_depth_bias) + tip_below
        return wait_z, lower_z

    def _scripted_hold_gripper(self) -> torch.Tensor:
        target = self._rod_radius - self._scripted_hold_margin
        value = self._gripper_action_for_aperture(2.0 * target)
        return torch.full((self.num_envs, 1), value, device=self.device, dtype=torch.float32)
