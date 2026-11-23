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


class ScriptedScanAimMixin(_ScriptedInterceptMixin):
    block: Actor
    _muzzle_ahead_of_tcp: float
    _handle_half_width: float
    _standoff_target: torch.Tensor

    _scripted_hold_margin: float = 0.0005

    _scripted_pos_step: float = 0.015
    _scripted_v_ee_max: float = 0.225
    _scripted_accel_step: float | None = 0.00375

    _scripted_auto_comove_headroom: float = 4.0

    _scripted_v_align: float = 0.15
    _scripted_max_lead: float = 0.30

    _park_standoff: float = 0.090

    _upstream_y_mag: float = 0.35

    _commit_along_in: float = 0.008
    _commit_along_out: float = 0.025
    _commit_z_in: float = 0.006
    _commit_z_out: float = 0.020
    _commit_aim_in_deg: float = 6.0
    _commit_aim_out_deg: float = 16.0

    _max_inward_step: float = 0.012

    @property
    def _scripted_rider(self) -> Actor:
        return self.block

    def _patch_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _muzzle(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patch_z = self._patch_frame()[0][:, 2]
        return patch_z, patch_z

    def _scripted_commit_window(self) -> tuple[float, float]:
        return self._commit_along_in, self._commit_along_out

    def _scripted_hold_gripper(self) -> torch.Tensor:
        target = self._handle_half_width - self._scripted_hold_margin
        value = self._gripper_action_for_aperture(2.0 * target)
        return torch.full((self.num_envs, 1), value, device=self.device, dtype=torch.float32)

    def _gripper_down_target_q(self) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

        base_q = super()._gripper_down_target_q()
        n = self._patch_frame()[1]
        heading = torch.atan2(-n[:, 1], -n[:, 0])
        half = 0.5 * (heading - 0.5 * math.pi * self.belt.direction_sign)
        q_yaw = torch.zeros_like(base_q)
        q_yaw[:, 0] = torch.cos(half)
        q_yaw[:, 3] = torch.sin(half)
        aim_q = quaternion_multiply(q_yaw, base_q)

        axis = self._muzzle()[1]
        v = torch.cross(axis, -n, dim=-1)
        s = torch.linalg.norm(v, dim=-1)
        c = (axis * -n).sum(dim=-1).clamp(-1.0, 1.0)
        half_fix = 0.5 * torch.atan2(s, c)
        unit = v / s.clamp(min=1e-6).unsqueeze(-1)
        q_fix = torch.cat([torch.cos(half_fix).unsqueeze(-1), unit * torch.sin(half_fix).unsqueeze(-1)], dim=-1)
        return quaternion_multiply(q_fix, aim_q)

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

        patch_centre, n, u, w = self._patch_frame()
        sign = self.belt.direction_sign
        speed = torch.abs(self.belt.velocity)
        motion_dir = torch.zeros((self.num_envs, 2), device=device)
        motion_dir[:, 1] = sign
        self._scripted_motion_dir = motion_dir

        station = patch_centre + n * self._park_standoff
        self._update_auto_speed_scale(tcp_pos, station)

        muzzle_centre, muzzle_axis = self._muzzle()
        rel_muzzle = muzzle_centre - patch_centre
        along = (rel_muzzle * u).sum(dim=-1) * sign
        e_w = (rel_muzzle * w).sum(dim=-1)

        aim_cos = -(muzzle_axis * n).sum(dim=-1)
        aim_deg = torch.rad2deg(torch.acos(aim_cos.clamp(-1.0, 1.0)))

        cross = (muzzle_centre[:, 0] - station[:, 0]).abs()
        align_gap = cross + self._scripted_lead_z_weight * e_w.abs()
        v_align = self._auto_scaled(self._scripted_v_align)
        max_lead = self._auto_scaled(self._scripted_max_lead)
        lead = speed * self._scripted_comove_lead_time + speed * align_gap / v_align
        if self._scripted_auto_scale is not None:
            near = (along.abs() < self._scripted_auto_bias_radius) & (cross < self._scripted_cross_scale)
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

        s_y = 1.0 - _smoothstep(
            (along.abs() - self._commit_along_in) / (self._commit_along_out - self._commit_along_in)
        )
        s_z = 1.0 - _smoothstep((e_w.abs() - self._commit_z_in) / (self._commit_z_out - self._commit_z_in))
        s_a = 1.0 - _smoothstep(
            (aim_deg - self._commit_aim_in_deg) / (self._commit_aim_out_deg - self._commit_aim_in_deg)
        )
        commit = s_y * s_z * s_a
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

        standoff = (rel_muzzle * n).sum(dim=-1)
        standoff_cmd = self._park_standoff + (self._standoff_target - self._park_standoff) * commit
        floor = torch.minimum(standoff - self._max_inward_step, torch.full_like(standoff, self._park_standoff))
        standoff_cmd = torch.maximum(standoff_cmd, floor)

        target_muzzle = patch_centre + n * standoff_cmd.unsqueeze(-1)
        target_muzzle[:, 1] = target_muzzle[:, 1] + sign * lead
        oy = self.workspace_offset[1]
        upstream = (target_muzzle[:, 1] - oy) * sign
        target_muzzle[:, 1] = oy + sign * upstream.clamp(min=-self._upstream_y_mag)
        target_world = target_muzzle - (muzzle_centre - tcp_pos)

        pos_input = self._position_input(target_world, tcp_pos)
        rot_input = self._rotation_input(tcp_q)
        gripper = self._scripted_hold_gripper()
        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
