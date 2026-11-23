from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@dataclass(frozen=True)
class _AimPose:
    p: torch.Tensor


class _LoadOrEscapeAim:
    def __init__(self, env: ScriptedClearArchMixin) -> None:
        self._env = env

    @property
    def pose(self) -> _AimPose:
        return _AimPose(self._env.scripted_aim_world_pos())


class ScriptedClearArchMixin(_ScriptedPlaceFamilyMixin):
    pallet: Actor
    _part_half_z: torch.Tensor
    _bin_mouth_local_z: float

    def arch_world_pos(self) -> torch.Tensor:
        raise NotImplementedError

    def arch_span_clearance(self, world: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    _scripted_accel_step: float | None = 0.0025

    _scripted_center_r_in: float = 0.012
    _scripted_center_r_out: float = 0.05
    _scripted_lower_xy_tol: float = 0.012
    _scripted_lower_z_tol: float = 0.010
    _scripted_release_steps: int = 4

    _scripted_wait_clearance: float = 0.030
    _scripted_release_rim_gap: float = 0.003

    _scripted_withdraw_rim_margin: float = 10.0
    _scripted_withdraw_z_above: float = 0.25

    _escape_dx: float = 0.20
    _escape_behind: float = 0.12
    _escape_clear_margin: float = 0.015

    def scripted_aim_world_pos(self) -> torch.Tensor:
        bin_pos = self.pallet.pose.p
        if not hasattr(self, "_scripted_phase"):
            return bin_pos
        is_withdraw = self._scripted_phase == self._PHASE_WITHDRAW
        return torch.where(is_withdraw.unsqueeze(-1), self._escape_world_pos(), bin_pos)

    def _escape_world_pos(self) -> torch.Tensor:
        ox, _oy, _oz = self.workspace_offset
        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        sign = self.belt.direction_sign

        escape = tcp_pos.clone()
        escape[:, 0] = ox - self._escape_dx

        clear_now = self.arch_span_clearance(tcp_pos) > self._escape_clear_margin
        want_y = self.arch_world_pos()[:, 1] - self._escape_behind * sign
        upstream = (want_y - tcp_pos[:, 1]) * sign < 0.0
        escape[:, 1] = torch.where(clear_now & upstream, want_y, tcp_pos[:, 1])
        return escape

    @property
    def _scripted_rider(self) -> Actor:
        if not hasattr(self, "_scripted_aim"):
            self._scripted_aim = _LoadOrEscapeAim(self)
        return cast("Actor", self._scripted_aim)

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return self.pallet.pose.p[:, 2] + self._bin_mouth_local_z

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mouth_z = self.pallet.pose.p[:, 2] + self._bin_mouth_local_z
        wait_z = mouth_z + self._scripted_wait_clearance + self._part_half_z
        lower_z = mouth_z + self._scripted_release_rim_gap + self._part_half_z
        return wait_z, lower_z

    def _scripted_release_gripper(self) -> torch.Tensor:
        open_val, _ = self._scripted_gripper_values()
        return torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32)
