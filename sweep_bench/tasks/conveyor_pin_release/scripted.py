from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

from sweep_bench.tasks.conveyor_base.scripted_families import ScriptedGraspMixin

if TYPE_CHECKING:
    import numpy as np
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedPinReleaseMixin(ScriptedGraspMixin):
    _PR_GRASP = 0
    _PR_WITHDRAW = 1
    _PR_LIFT = 2
    _PR_HOLD = 3

    carrier: Actor
    pin: Actor
    _bore_y: float
    _pin_grip_z: float
    _pin_post_radius: float
    _pin_clear_tip_x: float
    _tail_x_hi: float
    _pin_tip_overhang: float

    @property
    def _pin_grip_x(self) -> float:
        raise NotImplementedError

    def _pin_tip_local_x(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_approach_z_above: float = 0.15
    _scripted_open_finger_gap: float = 0.020
    _scripted_hold_margin: float = 0.004
    _pr_grip_z_above: float = 0.022
    _scripted_close_z_tol: float = 0.014
    _scripted_stopped_close_z_tol: float = 0.014

    _scripted_close_steps: int = 6

    _pr_grasp_check_dist: float = 0.05

    _pr_withdraw_rate: float = 0.010
    _pr_withdraw_rate_per_speed: float = 0.05
    _pr_withdraw_exit: float = 0.046
    _pr_withdraw_stroke: float = 0.056
    _pr_lead_margin: float = 0.008
    _pr_withdraw_max_steps: int = 26

    _pr_hold_z_above: float = 0.17
    _pr_hold_trigger_z: float = 0.14

    @property
    def _scripted_rider(self) -> Actor:
        return self.pin

    def _scripted_gripper_values(self) -> tuple[float, float]:
        open_val = self._gripper_action_for_aperture(self._scripted_open_finger_gap)
        return open_val, self._gripper_action_for_aperture(2.0 * (self._pin_post_radius - self._scripted_hold_margin))

    def _rest_aperture(self, env_idx: torch.Tensor) -> float | np.ndarray:
        return self._scripted_open_finger_gap

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_pr_phase"):
            self._pr_phase = torch.full((b,), self._PR_GRASP, dtype=torch.long, device=device)
            self._pr_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._pr_ramp = torch.zeros((b,), device=device)
            self._pr_hold_xy = torch.zeros((b, 2), device=device)
            self._pr_withdraw_z = torch.zeros((b,), device=device)
        if env_idx is None:
            self._pr_phase[:] = self._PR_GRASP
            self._pr_counter[:] = 0
            self._pr_ramp[:] = 0.0
        else:
            self._pr_phase[env_idx] = self._PR_GRASP
            self._pr_counter[env_idx] = 0
            self._pr_ramp[env_idx] = 0.0

    def _pr_along_belt_lead(self) -> torch.Tensor:
        sign = self.belt.direction_sign
        speed = torch.abs(self.belt.velocity)
        t = (
            (speed - self._scripted_auto_bias_v_lo) / (self._scripted_auto_bias_v_hi - self._scripted_auto_bias_v_lo)
        ).clamp(0.0, 1.0)
        bias_ramp = t * t * (3.0 - 2.0 * t)
        lead = speed * self._scripted_close_lead_time + self._scripted_auto_lead_bias * bias_ramp
        return sign * torch.minimum(lead, speed / self.control_freq + self._pr_lead_margin)

    def _pr_position_input(self, target_world: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
        cap = self._auto_scaled(self._scripted_pos_step)
        cap_t = torch.as_tensor(
            cap.unsqueeze(-1) if isinstance(cap, torch.Tensor) else cap,
            dtype=tcp_pos.dtype,
            device=tcp_pos.device,
        )
        delta = torch.clamp(target_world - tcp_pos, -cap_t, cap_t)
        return torch.clip(delta / 0.1, -1.0, 1.0)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_pr_phase"):
            self.reset_scripted()

        grasp_action = super().compute_scripted_action()

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        carrier_pos = self.carrier.pose.p
        rot = quaternion_to_matrix(self.carrier.pose.q)

        pr = self._pr_phase

        pin_in_hand = torch.linalg.norm(tcp_pos - self.pin.pose.p, dim=-1) < self._pr_grasp_check_dist
        to_withdraw = (pr == self._PR_GRASP) & (self._scripted_phase >= self._PHASE_LIFT) & pin_in_hand
        self._pr_withdraw_z = torch.where(to_withdraw, tcp_pos[:, 2], self._pr_withdraw_z)
        pr = torch.where(to_withdraw, torch.full_like(pr, self._PR_WITHDRAW), pr)

        in_withdraw = pr == self._PR_WITHDRAW
        self._pr_counter = torch.where(in_withdraw, self._pr_counter + 1, torch.zeros_like(self._pr_counter))
        rate = torch.clamp(self._pr_withdraw_rate_per_speed * torch.abs(self.belt.velocity), min=self._pr_withdraw_rate)
        self._pr_ramp = torch.where(
            in_withdraw, (self._pr_ramp + rate).clamp(max=self._pr_withdraw_stroke), self._pr_ramp
        )
        seated_tip_x = self._tail_x_hi + self._pin_tip_overhang
        retracted = seated_tip_x - self._pin_tip_local_x()
        to_lift = in_withdraw & (
            (retracted >= self._pr_withdraw_exit) | (self._pr_counter >= self._pr_withdraw_max_steps)
        )
        self._pr_hold_xy = torch.where(to_lift.unsqueeze(-1), tcp_pos[:, :2], self._pr_hold_xy)
        pr = torch.where(to_lift, torch.full_like(pr, self._PR_LIFT), pr)

        to_hold = (pr == self._PR_LIFT) & (tcp_pos[:, 2] > deck_z + self._pr_hold_trigger_z)
        pr = torch.where(to_hold, torch.full_like(pr, self._PR_HOLD), pr)
        self._pr_phase = pr

        target = tcp_pos.clone()

        anchor_local = torch.zeros_like(tcp_pos)
        anchor_local[:, 0] = self._pin_grip_x - self._pr_ramp
        anchor_local[:, 1] = self._bore_y
        anchor = carrier_pos + torch.bmm(rot, anchor_local.unsqueeze(-1)).squeeze(-1)

        is_withdraw = pr == self._PR_WITHDRAW
        target[:, 0] = torch.where(is_withdraw, anchor[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_withdraw, anchor[:, 1] + self._pr_along_belt_lead(), target[:, 1])
        target[:, 2] = torch.where(is_withdraw, self._pr_withdraw_z, target[:, 2])

        is_up = (pr == self._PR_LIFT) | (pr == self._PR_HOLD)
        target[:, 0] = torch.where(is_up, self._pr_hold_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_up, self._pr_hold_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_up, torch.full_like(target[:, 2], deck_z + self._pr_hold_z_above), target[:, 2])

        chained = pr != self._PR_GRASP
        pos_input = self._pr_position_input(target, tcp_pos).to(torch.float32)
        action = grasp_action.clone()
        action[:, :3] = torch.where(chained.unsqueeze(-1), pos_input, action[:, :3])

        cap = self._auto_scaled(self._scripted_pos_step)
        cap_t = torch.as_tensor(
            cap if isinstance(cap, torch.Tensor) else cap, dtype=tcp_pos.dtype, device=tcp_pos.device
        )
        seated_z = carrier_pos[:, 2] + self._pr_grip_z_above
        seated_dz = torch.clip(torch.clamp(seated_z - tcp_pos[:, 2], -cap_t, cap_t) / 0.1, -1.0, 1.0)
        is_descend = (pr == self._PR_GRASP) & (self._scripted_phase == self._PHASE_DESCEND)
        action[:, 2] = torch.where(is_descend, seated_dz.to(torch.float32), action[:, 2])
        is_close = (pr == self._PR_GRASP) & (self._scripted_phase == self._PHASE_CLOSE)
        action[:, 2] = torch.where(is_close, torch.zeros_like(action[:, 2]), action[:, 2])
        return action
