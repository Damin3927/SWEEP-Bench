from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedPullCordMixin(_ScriptedGraspFamilyMixin):
    _PC_APPROACHING = 0
    _PC_CLOSING = 1
    _PC_PULL = 2
    _PC_RELEASE = 3
    _PC_RETREAT = 4

    body: Actor
    handle: Actor

    def handle_travel(self) -> torch.Tensor:
        raise NotImplementedError

    def travel_latched(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_approach_z_above: float = 0.175
    _scripted_grasp_endgame_scale_floor: float = 1.0
    _scripted_close_xy_tol: float = 0.016
    _scripted_close_z_tol: float = 0.014
    _scripted_close_steps: int = 8
    _scripted_lift_z_above: float = 0.30
    _scripted_hold_z_above: float = 0.29

    _pc_stroke_target: float = 0.065
    _pc_pull_rate: float = 0.026
    _pc_lead_gain: float = 1.2
    _pc_lead_max: float = 0.15
    _pc_pull_max_steps: int = 24

    _pc_grip_finger_target: float = 0.002

    _pc_release_steps: int = 3
    _pc_retreat_z_above: float = 0.30

    def _scripted_gripper_values(self) -> tuple[float, float]:
        open_val, _full_close = super()._scripted_gripper_values()
        return open_val, self._gripper_action_for_aperture(2.0 * self._pc_grip_finger_target)

    @property
    def _scripted_rider(self) -> Actor:
        return self.handle

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_pc_phase"):
            self._pc_phase = torch.full((b,), self._PC_APPROACHING, dtype=torch.long, device=device)
            self._pc_pull_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._pc_release_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._pc_retreat_xy = torch.zeros((b, 2), device=device)
            self._pc_lead = torch.zeros((b,), device=device)
        if env_idx is None:
            self._pc_phase[:] = self._PC_APPROACHING
            self._pc_pull_counter[:] = 0
            self._pc_release_counter[:] = 0
            self._pc_lead[:] = 0.0
        else:
            self._pc_phase[env_idx] = self._PC_APPROACHING
            self._pc_pull_counter[env_idx] = 0
            self._pc_release_counter[env_idx] = 0
            self._pc_lead[env_idx] = 0.0

    def _pc_update_lead(self, active: torch.Tensor) -> None:
        d_sign = self.belt.direction_sign
        speed = torch.abs(self.belt.velocity)
        agent = cast("Panda", self.agent)
        v_along = agent.tcp.linear_velocity[:, 1] * d_sign
        trim = self._pc_lead + self._pc_lead_gain * (speed - v_along) / self.control_freq
        self._pc_lead = torch.where(active, trim.clamp(0.0, self._pc_lead_max), self._pc_lead)

    def _pc_seed_lead(self, entering: torch.Tensor) -> None:
        seed = (self._scripted_prev_delta[:, 1] * self.belt.direction_sign).clamp(0.0, self._pc_lead_max)
        self._pc_lead = torch.where(entering, seed, self._pc_lead)

    def _pc_track_xy(self) -> torch.Tensor:
        aim = self.body.pose.p[:, :2].clone()
        aim[:, 1] = aim[:, 1] + self.belt.direction_sign * self._pc_lead
        return aim

    def _pc_position_input(self, target_world: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
        cap = self._auto_scaled(self._scripted_pos_step)
        cap_t = torch.as_tensor(cap, dtype=tcp_pos.dtype, device=tcp_pos.device)
        if cap_t.ndim == 1:
            cap_t = cap_t.unsqueeze(-1)
        delta = target_world - tcp_pos
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap_t / xy_norm, max=1.0)
        z = torch.maximum(torch.minimum(delta[:, 2:3], cap_t), -cap_t)
        return torch.clip(torch.cat([xy, z], dim=-1) / 0.1, -1.0, 1.0)

    def compute_scripted_action(self) -> torch.Tensor:
        grasp_action = super().compute_scripted_action()
        if not hasattr(self, "_pc_phase"):
            self.reset_scripted()

        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        travel = self.handle_travel()
        phase = self._pc_phase

        to_close = (phase == self._PC_APPROACHING) & (self._scripted_phase >= self._PHASE_CLOSE)
        phase = torch.where(to_close, torch.full_like(phase, self._PC_CLOSING), phase)
        self._pc_seed_lead(to_close)

        to_pull = (phase == self._PC_CLOSING) & (self._scripted_phase >= self._PHASE_LIFT)
        phase = torch.where(to_pull, torch.full_like(phase, self._PC_PULL), phase)

        in_pull = phase == self._PC_PULL
        self._pc_update_lead(((phase == self._PC_CLOSING) | in_pull) & ~to_close)
        self._pc_pull_counter = torch.where(in_pull, self._pc_pull_counter + 1, self._pc_pull_counter)
        to_release = in_pull & (self.travel_latched() | (self._pc_pull_counter >= self._pc_pull_max_steps))
        self._pc_retreat_xy = torch.where(to_release.unsqueeze(-1), tcp_pos[:, :2], self._pc_retreat_xy)
        phase = torch.where(to_release, torch.full_like(phase, self._PC_RELEASE), phase)

        in_release = phase == self._PC_RELEASE
        self._pc_release_counter = torch.where(in_release, self._pc_release_counter + 1, self._pc_release_counter)
        to_retreat = in_release & (self._pc_release_counter >= self._pc_release_steps)
        phase = torch.where(to_retreat, torch.full_like(phase, self._PC_RETREAT), phase)
        self._pc_phase = phase

        target = tcp_pos.clone()

        is_close = phase == self._PC_CLOSING
        is_pull = phase == self._PC_PULL
        tracking = is_close | is_pull
        track_xy = self._pc_track_xy()
        stroke_step = (self._pc_stroke_target - travel).clamp(min=0.0, max=self._pc_pull_rate)
        target[:, 0] = torch.where(tracking, track_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(tracking, track_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_close, self.handle.pose.p[:, 2], target[:, 2])
        target[:, 2] = torch.where(is_pull, tcp_pos[:, 2] + stroke_step, target[:, 2])

        is_release = phase == self._PC_RELEASE
        is_retreat = phase == self._PC_RETREAT
        target[:, 0] = torch.where(is_retreat, self._pc_retreat_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_retreat, self._pc_retreat_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(
            is_retreat, torch.full_like(target[:, 2], deck_z + self._pc_retreat_z_above), target[:, 2]
        )

        chained = phase != self._PC_APPROACHING
        pos_input = torch.where(chained.unsqueeze(-1), self._pc_position_input(target, tcp_pos), grasp_action[:, :3])

        open_val, _close_val = self._scripted_gripper_values()
        opened = is_release | is_retreat
        gripper = torch.where(
            opened.unsqueeze(-1),
            torch.full_like(grasp_action[:, 6:7], open_val),
            grasp_action[:, 6:7],
        )
        return torch.cat([pos_input, grasp_action[:, 3:6], gripper], dim=-1)
