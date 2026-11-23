from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedTrayLiftMixin(_ScriptedGraspFamilyMixin):
    tray: Actor

    _scripted_approach_z_above: float = 0.18

    _scripted_close_xy_tol: float = 0.020
    _scripted_close_z_tol: float = 0.014

    _scripted_lift_z_above: float = 0.32

    _tl_lift_pos_step: float = 0.018
    _tl_lift_accel_step: float = 0.006

    @property
    def _scripted_rider(self) -> Actor:
        return self.tray

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        if not hasattr(self, "_tl_prev_delta"):
            self._tl_prev_delta = torch.zeros((self.num_envs, 3), device=self.device)
        if env_idx is None:
            self._tl_prev_delta[:] = 0.0
        else:
            self._tl_prev_delta[env_idx] = 0.0

    def compute_scripted_action(self) -> torch.Tensor:
        action = super().compute_scripted_action()

        gentle = (self._scripted_phase == self._PHASE_LIFT) | (self._scripted_phase == self._PHASE_HOLD)
        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        target_world = tcp_pos.clone()
        target_world[:, 2] = deck_z + self._scripted_lift_z_above

        pos_input = self._lift_position_input(target_world, tcp_pos, gentle)
        return torch.cat([torch.where(gentle.unsqueeze(-1), pos_input, action[:, :3]), action[:, 3:]], dim=-1)

    def _lift_position_input(
        self, target_world: torch.Tensor, tcp_pos: torch.Tensor, gentle: torch.Tensor
    ) -> torch.Tensor:
        prev = self._tl_prev_delta
        accel = self._tl_lift_accel_step
        delta = prev + ((target_world - tcp_pos) - prev).clamp(-accel, accel)
        cap = self._tl_lift_pos_step
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap / xy_norm, max=1.0)
        z = delta[:, 2:3].clamp(-cap, cap)
        clamped = torch.cat([xy, z], dim=-1)
        self._tl_prev_delta = torch.where(gentle.unsqueeze(-1), clamped, self._scripted_prev_delta)
        return torch.clip(self._apply_reach_guard(clamped, tcp_pos) / 0.1, -1.0, 1.0)
