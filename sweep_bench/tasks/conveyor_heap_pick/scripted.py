from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedHeapPickMixin(_ScriptedGraspFamilyMixin):
    corral: Actor
    riser: Actor
    _riser_half_z: float

    def corral_clear_world_z(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_approach_z_above: float = 0.16

    _scripted_xy_align_tol: float = 0.012
    _scripted_close_xy_tol: float = 0.015
    _scripted_close_z_tol: float = 0.018
    _scripted_close_steps: int = 6

    _scripted_lift_z_above: float = 0.22
    _scripted_hold_z_above: float = 0.19

    _hp_extract_enabled: bool = True
    _hp_clear_margin: float = 0.006
    _hp_extract_scale_floor: float = 5.0

    @property
    def _scripted_rider(self) -> Actor:
        return self.riser

    def _extract_lead_bias(self) -> torch.Tensor:
        if self._scripted_auto_scale is None:
            return torch.zeros_like(self.belt.velocity)
        lo, hi = self._scripted_auto_bias_v_lo, self._scripted_auto_bias_v_hi
        t = ((torch.abs(self.belt.velocity) - lo) / (hi - lo)).clamp(0.0, 1.0)
        ramp = t * t * (3.0 - 2.0 * t)
        return self.belt.direction_sign * self._scripted_auto_lead_bias * ramp

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        if not hasattr(self, "_hp_cleared"):
            self._hp_cleared = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._hp_extracting = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._hp_track_offset = torch.zeros((self.num_envs, 2), device=self.device)
        if env_idx is None:
            self._hp_cleared[:] = False
            self._hp_extracting[:] = False
            self._hp_track_offset[:] = 0.0
        else:
            self._hp_cleared[env_idx] = False
            self._hp_extracting[env_idx] = False
            self._hp_track_offset[env_idx] = 0.0

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_hp_cleared"):
            self.reset_scripted()

        saved_delta, saved_velrand = self._scripted_prev_delta.clone(), self._velrand_x.clone()

        grasp_action = super().compute_scripted_action()
        if not self._hp_extract_enabled:
            return grasp_action

        grasp_delta, grasp_velrand = self._scripted_prev_delta.clone(), self._velrand_x.clone()
        self._scripted_prev_delta, self._velrand_x = saved_delta, saved_velrand

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        corral_xy = self.corral.pose.p[:, :2]
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        phase = self._scripted_phase
        lifting = (phase == self._PHASE_LIFT) | (phase == self._PHASE_HOLD)

        entering = lifting & ~self._hp_extracting
        self._hp_track_offset = torch.where(entering.unsqueeze(-1), tcp_pos[:, :2] - corral_xy, self._hp_track_offset)
        self._hp_extracting = self._hp_extracting | lifting

        clear_z = self.corral_clear_world_z()
        self._hp_cleared = self._hp_cleared | (
            (self.riser.pose.p[:, 2] - self._riser_half_z) > clear_z + self._hp_clear_margin
        )

        in_extract = lifting & ~self._hp_cleared

        if self._scripted_auto_scale is not None:
            self._scripted_auto_scale = torch.where(
                in_extract,
                self._scripted_auto_scale.clamp(min=self._hp_extract_scale_floor),
                self._scripted_auto_scale,
            )

        comove = torch.zeros_like(corral_xy)
        comove[:, 1] = self.belt.velocity * self._scripted_close_lead_time + self._extract_lead_bias()
        target = torch.zeros_like(tcp_pos)
        target[:, :2] = corral_xy + self._hp_track_offset + comove
        target[:, 2] = deck_z + self._scripted_lift_z_above

        pos_input = self._position_input(target, tcp_pos)
        rot_input = self._rotation_input(tcp_q)
        _open_val, close_val = self._scripted_gripper_values()
        gripper = torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)
        extract_action = torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)

        mask = in_extract.unsqueeze(-1)
        self._scripted_prev_delta = torch.where(mask, self._scripted_prev_delta, grasp_delta)
        self._velrand_x = torch.where(mask, self._velrand_x, grasp_velrand)
        return torch.where(mask, extract_action, grasp_action)
