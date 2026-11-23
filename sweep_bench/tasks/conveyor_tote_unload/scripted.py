from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedToteUnloadMixin(_ScriptedGraspFamilyMixin):
    part: Actor
    tote: Actor
    _part_half_z: torch.Tensor

    def tote_mouth_world_z(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_approach_z_above: float = 0.15

    _scripted_xy_align_tol: float = 0.012
    _scripted_close_xy_tol: float = 0.015
    _scripted_close_z_tol: float = 0.018
    _scripted_close_steps: int = 6

    _scripted_lift_z_above: float = 0.24
    _scripted_hold_z_above: float = 0.21

    _tu_clear_margin: float = 0.006
    _tu_extract_scale_floor: float = 5.0

    @property
    def _scripted_rider(self) -> Actor:
        return self.part

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        if not hasattr(self, "_tu_cleared"):
            self._tu_cleared = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._tu_extracting = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._tu_track_offset = torch.zeros((self.num_envs, 2), device=self.device)
        if env_idx is None:
            self._tu_cleared[:] = False
            self._tu_extracting[:] = False
            self._tu_track_offset[:] = 0.0
        else:
            self._tu_cleared[env_idx] = False
            self._tu_extracting[env_idx] = False
            self._tu_track_offset[env_idx] = 0.0

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_tu_cleared"):
            self.reset_scripted()

        grasp_action = super().compute_scripted_action()

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        tote_xy = self.tote.pose.p[:, :2]
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        phase = self._scripted_phase
        lifting = (phase == self._PHASE_LIFT) | (phase == self._PHASE_HOLD)

        entering = lifting & ~self._tu_extracting
        self._tu_track_offset = torch.where(entering.unsqueeze(-1), tcp_pos[:, :2] - tote_xy, self._tu_track_offset)
        self._tu_extracting = self._tu_extracting | lifting

        mouth_z = self.tote_mouth_world_z()
        self._tu_cleared = self._tu_cleared | (
            (self.part.pose.p[:, 2] - self._part_half_z) > mouth_z + self._tu_clear_margin
        )

        in_extract = lifting & ~self._tu_cleared

        if self._scripted_auto_scale is not None:
            self._scripted_auto_scale = torch.where(
                in_extract,
                self._scripted_auto_scale.clamp(min=self._tu_extract_scale_floor),
                self._scripted_auto_scale,
            )

        comove = torch.zeros_like(tote_xy)
        comove[:, 1] = self.belt.velocity * self._scripted_close_lead_time
        target = torch.zeros_like(tcp_pos)
        target[:, :2] = tote_xy + self._tu_track_offset + comove
        target[:, 2] = deck_z + self._scripted_lift_z_above

        pos_input = self._position_input(target, tcp_pos)
        rot_input = self._rotation_input(tcp_q)
        _open_val, close_val = self._scripted_gripper_values()
        gripper = torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)
        extract_action = torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)

        return torch.where(in_extract.unsqueeze(-1), extract_action, grasp_action)
