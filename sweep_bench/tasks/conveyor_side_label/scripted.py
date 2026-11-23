from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    _ScriptedInterceptMixin,
    _smoothstep,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor
    from mani_skill.utils.structs.pose import Pose


class ScriptedSideLabelMixin(_ScriptedInterceptMixin):
    box: Actor
    _patch_pose: Pose
    _pad_half_y: float
    _pad_z_offset: float
    _face_ahead_of_tcp: float

    _scripted_hold_margin: float = 0.0005

    _scripted_pos_step: float = 0.015
    _scripted_v_ee_max: float = 0.225

    _scripted_v_align: float = 0.15
    _scripted_max_lead: float = 0.30

    _standoff_gap: float = 0.006
    _press_penetration: float = 0.006

    _commit_y_in: float = 0.006
    _commit_y_out: float = 0.020
    _commit_z_in: float = 0.002
    _commit_z_out: float = 0.008

    _max_lateral_step: float = 0.004
    _lateral_guard_gap: float = 0.012

    @property
    def _scripted_rider(self) -> Actor:
        return self.box

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        press_z = (self.box.pose * self._patch_pose).p[:, 2] - self._pad_z_offset
        return press_z, press_z

    def _scripted_commit_window(self) -> tuple[float, float]:
        return self._commit_y_in, self._commit_y_out

    def _scripted_hold_gripper(self) -> torch.Tensor:
        target = self._pad_half_y - self._scripted_hold_margin
        value = self._gripper_action_for_aperture(2.0 * target)
        return torch.full((self.num_envs, 1), value, device=self.device, dtype=torch.float32)

    def _gripper_down_target_q(self) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

        base_q = super()._gripper_down_target_q()
        half = -0.25 * math.pi * self.belt.direction_sign
        q_yaw = torch.zeros_like(base_q)
        q_yaw[:, 0] = torch.cos(half)
        q_yaw[:, 3] = torch.sin(half)
        return quaternion_multiply(q_yaw, base_q)

    def _clamp_pos_delta(self, delta_pos: torch.Tensor) -> torch.Tensor:
        from sweep_bench.tasks.conveyor_base.scripted import _as_col

        accel_step = _as_col(self._auto_scaled(self._scripted_accel_step))
        if accel_step is not None:
            prev = self._scripted_prev_delta
            accel = torch.as_tensor(accel_step, dtype=delta_pos.dtype, device=delta_pos.device)
            delta_pos = prev + (delta_pos - prev).clamp(-accel, accel)
        pos_step = torch.as_tensor(
            _as_col(self._auto_scaled(self._scripted_pos_step)),
            dtype=delta_pos.dtype,
            device=delta_pos.device,
        )
        cap = pos_step * self._advance_velrand_step() if self.velocity_randomization.enabled else pos_step
        clamped = torch.clamp(delta_pos, -cap, cap)
        self._scripted_prev_delta = clamped
        return clamped

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        self._reset_scripted_shared(env_idx)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_scripted_motion_dir"):
            self.reset_scripted()

        device = self.device
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        tcp_vel = agent.tcp.linear_velocity

        patch_world = (self.box.pose * self._patch_pose).p
        sign = self.belt.direction_sign
        speed = torch.abs(self.belt.velocity)
        motion_dir = torch.zeros((self.num_envs, 2), device=device)
        motion_dir[:, 1] = sign
        self._scripted_motion_dir = motion_dir

        face_x = patch_world[:, 0]
        press_z = patch_world[:, 2] - self._pad_z_offset
        stand_x = face_x - self._face_ahead_of_tcp - self._standoff_gap
        press_x = face_x - self._face_ahead_of_tcp + self._press_penetration

        station = torch.stack([stand_x, patch_world[:, 1], press_z], dim=-1)
        self._update_auto_speed_scale(tcp_pos, station)

        rel_x = tcp_pos[:, 0] - stand_x
        rel_z = tcp_pos[:, 2] - press_z
        along = (tcp_pos[:, 1] - patch_world[:, 1]) * sign
        approach_gap = (stand_x - tcp_pos[:, 0]).clamp(min=0.0)
        align_gap = approach_gap + self._scripted_lead_z_weight * rel_z.abs()
        v_align = self._auto_scaled(self._scripted_v_align)
        max_lead = self._auto_scaled(self._scripted_max_lead)
        lead = speed * self._scripted_comove_lead_time + speed * align_gap / v_align
        if self._scripted_auto_scale is not None:
            near = (along.abs() < self._scripted_auto_bias_radius) & (rel_x.abs() < self._scripted_cross_scale)
            bias = torch.where(
                near,
                (self._scripted_auto_lead_bias - self._scripted_auto_bias_gain * along).clamp(
                    0.0, self._scripted_auto_bias_max
                ),
                self._scripted_auto_lead_bias,
            )
            self._scripted_auto_lead_bias = bias
            lead = lead + bias
        lead = lead.clamp(max=max_lead)
        aim_y = patch_world[:, 1] + sign * lead

        y_err = torch.abs(along)
        s_y = 1.0 - _smoothstep((y_err - self._commit_y_in) / (self._commit_y_out - self._commit_y_in))
        s_z = 1.0 - _smoothstep((rel_z.abs() - self._commit_z_in) / (self._commit_z_out - self._commit_z_in))
        commit = s_y * s_z
        if self._scripted_auto_scale is not None:
            slip = torch.abs(tcp_vel[:, 1] * sign - speed)
            crossing = _smoothstep(
                (slip - self._scripted_auto_dip_slip_lo)
                / (self._scripted_auto_dip_slip_hi - self._scripted_auto_dip_slip_lo)
            )
            park = _smoothstep(
                (speed - self._scripted_auto_park_v_lo)
                / (self._scripted_auto_park_v_hi - self._scripted_auto_park_v_lo)
            )
            commit = commit * (1.0 - park * crossing)

        target_x = stand_x + (press_x - stand_x) * commit
        dx = target_x - tcp_pos[:, 0]
        guarded = tcp_pos[:, 0] > stand_x - self._lateral_guard_gap
        dx = torch.where(guarded, dx.clamp(-self._max_lateral_step, self._max_lateral_step), dx)
        target_world = torch.stack([tcp_pos[:, 0] + dx, aim_y, press_z], dim=-1)

        pos_input = self._position_input(target_world, tcp_pos)
        rot_input = self._rotation_input(tcp_q)
        gripper = self._scripted_hold_gripper()
        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
