from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedReachMixin(_ScriptedInsertFamilyMixin):
    marker: Actor
    _target_height: float
    _reach_tol: float

    _scripted_pos_step: float = 0.015
    _scripted_v_ee_max: float = 0.225
    _scripted_accel_step: float | None = 0.00375

    _scripted_v_align: float = 0.15
    _scripted_max_lead: float = 0.30
    _scripted_cross_scale: float = 0.05
    _scripted_prox_scale: float = 0.05

    @property
    def _scripted_rider(self) -> Actor:
        return self.marker

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._reach_tol

    @property
    def _scripted_socket_chamfer(self) -> float:
        return 0.0

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_z = rider_pos[:, 2] + self._target_height
        return target_z, target_z

    def _scripted_hold_gripper(self) -> torch.Tensor:
        _, close_val = self._scripted_gripper_values()
        return torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)
