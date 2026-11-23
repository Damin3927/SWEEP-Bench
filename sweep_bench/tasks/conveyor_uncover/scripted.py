from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedUncoverMixin(_ScriptedGraspFamilyMixin):
    lid: Actor
    bin: Actor
    _bin_height: float
    _bin_origin_above_deck: float
    _lid_origin_to_plug_bottom: float

    _scripted_approach_z_above: float = 0.13

    _scripted_close_xy_tol: float = 0.015
    _scripted_close_z_tol: float = 0.015

    _scripted_lift_z_above: float = 0.24
    _scripted_hold_z_above: float = 0.22

    _scripted_extract_clearance: float = 0.005

    @property
    def _scripted_rider(self) -> Actor:
        return self.lid

    def compute_scripted_action(self) -> torch.Tensor:
        action = super().compute_scripted_action()

        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        bin_pos = self.bin.pose.p
        lid_pos = self.lid.pose.p

        plug_bottom_z = lid_pos[:, 2] - self._lid_origin_to_plug_bottom
        rim_z = bin_pos[:, 2] + (self._bin_height - self._bin_origin_above_deck)
        extracting = (self._scripted_phase == self._PHASE_LIFT) & (
            plug_bottom_z < rim_z + self._scripted_extract_clearance
        )

        comove_y = self.belt.velocity * self._scripted_close_lead_time
        target = torch.stack(
            [
                bin_pos[:, 0],
                bin_pos[:, 1] + comove_y,
                torch.full_like(bin_pos[:, 2], deck_z + self._scripted_lift_z_above),
            ],
            dim=-1,
        )
        delta = target - tcp_pos
        cap = torch.as_tensor(self._auto_scaled(self._scripted_pos_step), device=delta.device, dtype=delta.dtype)
        cap = cap.reshape(-1, 1) if cap.ndim else cap.reshape(1, 1)
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap / xy_norm, max=1.0)
        z = torch.maximum(torch.minimum(delta[:, 2:3], cap), -cap)
        pos_input = torch.clip(torch.cat([xy, z], dim=-1) / 0.1, -1.0, 1.0)

        extract_action = torch.cat([pos_input.to(torch.float32), action[:, 3:]], dim=-1)
        return torch.where(extracting.unsqueeze(-1), extract_action, action)
