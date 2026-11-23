from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

from sweep_bench.tasks.conveyor_base.scripted_families import ScriptedGraspMixin

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedKeySlotMixin(ScriptedGraspMixin):
    _KS_GRASP = 0
    _KS_SLIDE = 1
    _KS_LIFT = 2

    carrier: Actor
    part: Actor
    _seated_x: float
    _open_x: float
    _main_keyhole_y: float
    _slide_len: float

    def _part_head_local(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_approach_z_above: float = 0.16
    _scripted_close_steps: int = 6

    _ks_grasp_check_dist: float = 0.06

    _ks_slide_rate: float = 0.010
    _ks_slide_rate_per_speed: float = 0.06
    _ks_slide_exit_tol: float = 0.004
    _ks_slide_max_steps: int = 24
    _ks_lead_gain: float = 0.3
    _ks_lead_max: float = 0.09

    _ks_lift_z_above: float = 0.10
    _ks_lift_z_cap: float = 0.020

    @property
    def _scripted_rider(self) -> Actor:
        return self.part

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_ks_phase"):
            self._ks_phase = torch.full((b,), self._KS_GRASP, dtype=torch.long, device=device)
            self._ks_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._ks_ramp = torch.zeros((b,), device=device)
            self._ks_slide_z = torch.zeros((b,), device=device)
            self._ks_lead_bias = torch.zeros((b,), device=device)
        if env_idx is None:
            self._ks_phase[:] = self._KS_GRASP
            self._ks_counter[:] = 0
            self._ks_ramp[:] = 0.0
            self._ks_lead_bias[:] = 0.0
        else:
            self._ks_phase[env_idx] = self._KS_GRASP
            self._ks_counter[env_idx] = 0
            self._ks_ramp[env_idx] = 0.0
            self._ks_lead_bias[env_idx] = 0.0

    def _ks_lead(self, anchor_y: torch.Tensor, part_y: torch.Tensor, chained: torch.Tensor) -> torch.Tensor:
        sign = self.belt.direction_sign
        along_err = (anchor_y - part_y) * sign
        bias = (self._ks_lead_bias + self._ks_lead_gain * along_err).clamp(-self._ks_lead_max, self._ks_lead_max)
        self._ks_lead_bias = torch.where(chained, bias, self._ks_lead_bias)
        return sign * self._ks_lead_bias

    def _ks_position_input(self, target_world: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
        cap = self._auto_scaled(self._scripted_pos_step)
        cap_t = torch.as_tensor(
            cap.unsqueeze(-1) if isinstance(cap, torch.Tensor) else cap,
            dtype=tcp_pos.dtype,
            device=tcp_pos.device,
        )
        z_cap = torch.clamp(cap_t, max=self._ks_lift_z_cap)
        raw = target_world - tcp_pos
        delta = torch.cat(
            [torch.clamp(raw[:, :2], -cap_t, cap_t), torch.clamp(raw[:, 2:3], -z_cap, z_cap)],
            dim=-1,
        )
        return torch.clip(delta / 0.1, -1.0, 1.0)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_ks_phase"):
            self.reset_scripted()

        grasp_action = super().compute_scripted_action()

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        carrier_pos = self.carrier.pose.p
        rot = quaternion_to_matrix(self.carrier.pose.q)

        ks = self._ks_phase

        part_in_hand = torch.linalg.norm(tcp_pos - self.part.pose.p, dim=-1) < self._ks_grasp_check_dist
        to_slide = (ks == self._KS_GRASP) & (self._scripted_phase >= self._PHASE_LIFT) & part_in_hand
        self._ks_slide_z = torch.where(to_slide, tcp_pos[:, 2], self._ks_slide_z)
        seed_lead = (
            torch.abs(self.belt.velocity) * self._scripted_close_lead_time + self._scripted_auto_lead_bias
        ).clamp(-self._ks_lead_max, self._ks_lead_max)
        self._ks_lead_bias = torch.where(to_slide, seed_lead, self._ks_lead_bias)
        ks = torch.where(to_slide, torch.full_like(ks, self._KS_SLIDE), ks)

        in_slide = ks == self._KS_SLIDE
        self._ks_counter = torch.where(in_slide, self._ks_counter + 1, torch.zeros_like(self._ks_counter))
        rate = torch.clamp(self._ks_slide_rate_per_speed * torch.abs(self.belt.velocity), min=self._ks_slide_rate)
        self._ks_ramp = torch.where(in_slide, (self._ks_ramp + rate).clamp(max=self._slide_len), self._ks_ramp)
        head_off = torch.abs(self._part_head_local()[:, 0] - self._open_x)
        to_lift = in_slide & ((head_off <= self._ks_slide_exit_tol) | (self._ks_counter >= self._ks_slide_max_steps))
        ks = torch.where(to_lift, torch.full_like(ks, self._KS_LIFT), ks)
        self._ks_phase = ks

        target = tcp_pos.clone()

        anchor_local = torch.zeros_like(tcp_pos)
        anchor_local[:, 0] = torch.where(
            ks == self._KS_SLIDE,
            self._seated_x - self._ks_ramp,
            torch.full_like(self._ks_ramp, self._open_x),
        )
        anchor_local[:, 1] = self._main_keyhole_y
        anchor = carrier_pos + torch.bmm(rot, anchor_local.unsqueeze(-1)).squeeze(-1)

        chained = ks != self._KS_GRASP
        lead = self._ks_lead(anchor[:, 1], self.part.pose.p[:, 1], chained)
        target[:, 0] = torch.where(chained, anchor[:, 0], target[:, 0])
        target[:, 1] = torch.where(chained, anchor[:, 1] + lead, target[:, 1])
        target[:, 2] = torch.where(in_slide, self._ks_slide_z, target[:, 2])
        target[:, 2] = torch.where(
            ks == self._KS_LIFT, torch.full_like(target[:, 2], deck_z + self._ks_lift_z_above), target[:, 2]
        )

        pos_input = self._ks_position_input(target, tcp_pos).to(torch.float32)
        action = grasp_action.clone()
        action[:, :3] = torch.where(chained.unsqueeze(-1), pos_input, action[:, :3])

        is_close = (ks == self._KS_GRASP) & (self._scripted_phase == self._PHASE_CLOSE)
        action[:, 2] = torch.where(is_close, torch.zeros_like(action[:, 2]), action[:, 2])
        return action
