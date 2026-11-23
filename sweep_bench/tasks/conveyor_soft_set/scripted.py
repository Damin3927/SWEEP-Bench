from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedSoftSetMixin(_ScriptedPlaceFamilyMixin):
    carrier: Actor
    _part_half_z: torch.Tensor
    _platform_top_above_carrier: float

    _scripted_accel_step: float | None = 0.0025

    _scripted_center_r_in: float = 0.015
    _scripted_center_r_out: float = 0.05
    _scripted_lower_xy_tol: float = 0.010
    _scripted_lower_z_tol: float = 0.006
    _scripted_release_steps: int = 6

    _scripted_wait_clearance: float = 0.012
    _scripted_release_gap: float = 0.002

    _ss_release_max_steps: int = 30

    @property
    def _scripted_rider(self) -> Actor:
        return self.carrier

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        if not hasattr(self, "_ss_release_wait"):
            self._ss_release_wait = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
            self._ss_released = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        if env_idx is None:
            self._ss_release_wait[:] = 0
            self._ss_released[:] = False
        else:
            self._ss_release_wait[env_idx] = 0
            self._ss_released[env_idx] = False

    def compute_scripted_action(self) -> torch.Tensor:
        action = super().compute_scripted_action()
        if not hasattr(self, "_ss_release_wait"):
            self.reset_scripted()

        agent = cast("Panda", self.agent)
        sign = self.belt.direction_sign
        slip = torch.abs(agent.tcp.linear_velocity[:, 1] * sign - torch.abs(self.belt.velocity))
        fresh = (self._scripted_phase == self._PHASE_RELEASE) & ~self._ss_released
        hold = fresh & (slip > self._scripted_auto_dip_slip_lo) & (self._ss_release_wait < self._ss_release_max_steps)
        self._ss_release_wait = torch.where(hold, self._ss_release_wait + 1, self._ss_release_wait)
        self._ss_released = self._ss_released | (fresh & ~hold)

        self._scripted_phase = torch.where(
            hold, torch.full_like(self._scripted_phase, self._PHASE_INTERCEPT), self._scripted_phase
        )
        self._scripted_release_counter = torch.where(
            hold, torch.zeros_like(self._scripted_release_counter), self._scripted_release_counter
        )
        _, close_val = self._scripted_gripper_values()
        action[:, 6] = torch.where(hold, torch.full_like(action[:, 6], close_val), action[:, 6])
        return action

    def _platform_surface_z(self, carrier_pos: torch.Tensor) -> torch.Tensor:
        return carrier_pos[:, 2] + self._platform_top_above_carrier

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return self._platform_surface_z(rider_pos) + 2.0 * self._part_half_z

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        surface_z = self._platform_surface_z(rider_pos)
        wait_z = surface_z + self._scripted_wait_clearance + self._part_half_z
        lower_z = surface_z + self._scripted_release_gap + self._part_half_z
        return wait_z, lower_z

    def _scripted_release_gripper(self) -> torch.Tensor:
        open_val, _ = self._scripted_gripper_values()
        return torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32)
