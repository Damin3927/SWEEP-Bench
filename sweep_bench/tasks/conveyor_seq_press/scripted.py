from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


@dataclass(frozen=True)
class _AimPoint:
    pose: Pose


class ScriptedSeqPressMixin(_ScriptedInsertFamilyMixin):
    caps: list[Actor]
    _body_origin_above_deck: float
    _cap_half_height: float
    _cap_rest_gap: float
    _first_cap: torch.Tensor
    _second_cap: torch.Tensor
    _first_latched: torch.Tensor

    _scripted_accel_step: float | None = 0.005

    _scripted_wait_clearance: float = 0.02
    _scripted_press_overshoot: float = 0.01

    _scripted_commit_half_width: float = 0.010
    _scripted_commit_chamfer: float = 0.002

    @property
    def _scripted_rider(self) -> Actor:
        caps_p = torch.stack([cap.pose.p for cap in self.caps], dim=1)
        idx = torch.where(self._first_latched, self._second_cap, self._first_cap)
        target = caps_p.gather(1, idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
        return cast("Actor", _AimPoint(Pose.create_from_pq(target)))

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._scripted_commit_half_width

    @property
    def _scripted_socket_chamfer(self) -> float:
        return self._scripted_commit_chamfer

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, float]:
        cap_pressed_top = deck_z + self._body_origin_above_deck + 2.0 * self._cap_half_height
        cap_rest_top = cap_pressed_top + self._cap_rest_gap
        wait_z = cap_rest_top + self._scripted_wait_clearance
        lower_z = cap_pressed_top - self._scripted_press_overshoot
        return wait_z, lower_z

    def _scripted_hold_gripper(self) -> torch.Tensor:
        _, close_val = self._scripted_gripper_values()
        return torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)
