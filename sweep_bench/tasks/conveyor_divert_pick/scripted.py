from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedDivertPickMixin(_ScriptedGraspFamilyMixin):
    target: Actor
    _plough_half_length: float

    def rail_dir(self, sign: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def rail_offset(self, xy: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def lane_diversion(self, xy: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    _scripted_approach_z_above: float = 0.14

    _scripted_settle_max_steps: int = 4
    _scripted_close_steps: int = 8
    _scripted_lift_z_above: float = 0.24
    _scripted_hold_z_above: float = 0.21

    _scripted_min_diversion: float = 0.05

    _plough_speed_ratio: float = 1.051

    @property
    def _scripted_rider(self) -> Actor:
        return self.target

    def _rail_motion_dir(self) -> torch.Tensor:
        return self.rail_dir(self.belt.direction_sign)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        rail = self._rail_motion_dir()
        if env_idx is None:
            self._scripted_motion_dir[:] = rail
        else:
            self._scripted_motion_dir[env_idx] = rail[env_idx]

    def _update_motion_dir(self, v_xy: torch.Tensor, update_mask: torch.Tensor) -> None:
        rail = self._rail_motion_dir()
        self._scripted_motion_dir = torch.where(update_mask.unsqueeze(-1), rail, self._scripted_motion_dir)

    def _divert_comove_lead(self) -> torch.Tensor:
        from sweep_bench.tasks.conveyor_base.scripted_families import _smoothstep

        sign = self.belt.direction_sign
        v = self.belt.velocity
        on_rail = torch.abs(self.rail_offset(self.target.pose.p[:, :2], sign)) <= self._plough_half_length

        along = torch.abs(v) * self._scripted_close_lead_time
        if self._scripted_auto_scale is not None:
            speed_ramp = _smoothstep(
                (torch.abs(v) - self._scripted_auto_bias_v_lo)
                / (self._scripted_auto_bias_v_hi - self._scripted_auto_bias_v_lo)
            )
            along = along + self._scripted_auto_lead_bias * speed_ramp

        rail = self.rail_dir(sign)
        rail_lead = rail * (along * self._plough_speed_ratio).unsqueeze(-1)
        belt_lead = torch.zeros_like(rail_lead)
        belt_lead[:, 1] = along * sign
        return torch.where(on_rail.unsqueeze(-1), rail_lead, belt_lead)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        saved_delta, saved_velrand = self._scripted_prev_delta.clone(), self._velrand_x.clone()

        grasp_action = super().compute_scripted_action()
        grasp_delta, grasp_velrand = self._scripted_prev_delta.clone(), self._velrand_x.clone()
        self._scripted_prev_delta, self._velrand_x = saved_delta, saved_velrand

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        rider_pos = self.target.pose.p
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        held_back = (self.lane_diversion(rider_pos[:, :2]) < self._scripted_min_diversion) & (
            (self._scripted_phase == self._PHASE_DESCEND) | (self._scripted_phase == self._PHASE_CLOSE)
        )
        self._scripted_phase = torch.where(
            held_back, torch.full_like(self._scripted_phase, self._PHASE_APPROACH), self._scripted_phase
        )
        self._scripted_close_counter = torch.where(
            held_back, torch.zeros_like(self._scripted_close_counter), self._scripted_close_counter
        )

        phase = self._scripted_phase
        is_approach = phase == self._PHASE_APPROACH
        is_descend = phase == self._PHASE_DESCEND
        is_close = phase == self._PHASE_CLOSE
        tracking = is_approach | is_descend | is_close

        xy_dist = torch.linalg.norm(tcp_pos[:, :2] - rider_pos[:, :2], dim=-1)
        look = self._lookahead_xy(self.target.linear_velocity[:, :2], xy_dist, rider_pos[:, :2] - tcp_pos[:, :2])
        look = torch.where(is_close.unsqueeze(-1), torch.zeros_like(look), look)
        lead = self._divert_comove_lead()

        target = torch.zeros_like(tcp_pos)
        target[:, :2] = rider_pos[:, :2] + look + lead
        target[:, 2] = torch.where(
            is_approach,
            torch.full_like(rider_pos[:, 2], deck_z + self._scripted_approach_z_above),
            rider_pos[:, 2],
        )

        pos_input = self._position_input(target, tcp_pos)
        rot_input = self._rotation_input(tcp_q)
        open_val, close_val = self._scripted_gripper_values()
        gripper = torch.where(
            is_approach | is_descend,
            torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32),
            torch.full((self.num_envs,), close_val, device=self.device, dtype=torch.float32),
        ).unsqueeze(-1)
        local_action = torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)

        mask = tracking.unsqueeze(-1)
        self._scripted_prev_delta = torch.where(mask, self._scripted_prev_delta, grasp_delta)
        self._velrand_x = torch.where(mask, self._velrand_x, grasp_velrand)
        return torch.where(mask, local_action, grasp_action)
