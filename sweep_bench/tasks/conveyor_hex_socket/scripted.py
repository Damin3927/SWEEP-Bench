from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)
from sweep_bench.tasks.conveyor_hex_socket.assets import fold_hex

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor
    from mani_skill.utils.structs.pose import Pose

    from sweep_bench.tasks.conveyor_base.scripted_families import _InterceptState


class ScriptedHexSocketMixin(_ScriptedInsertFamilyMixin):
    plate: Actor
    bit: Actor
    _bit_tip_offset: Pose
    _plate_half_height: float
    _chamfer: float
    _insert_depth: float
    _bit_grasp_offset: float
    _bit_half_length: float
    _bit_half_af: float
    _mouth_half_af: float
    _clear_lateral: float
    _socket_yaw: torch.Tensor
    _scripted_socket_yaw_cmd: torch.Tensor

    _scripted_pos_step: float = 0.015
    _scripted_v_ee_max: float = 0.225
    _scripted_accel_step: float | None = 0.00375

    _scripted_v_align: float = 0.15
    _scripted_max_lead: float = 0.30
    _scripted_cross_scale: float = 0.05
    _scripted_prox_scale: float = 0.05

    _scripted_rot_gain: float = 0.6

    _scripted_wait_clearance: float = 0.010
    _scripted_seat_depth_margin: float = 0.006

    _scripted_hold_margin: float = 0.0045
    _scripted_hold_init_margin: float = 0.0005

    _scripted_commit_hold_scale: float = 2.5

    @property
    def _scripted_rider(self) -> Actor:
        return self.plate

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        if not hasattr(self, "_scripted_committed"):
            self._scripted_committed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if env_idx is None:
            self._scripted_committed[:] = False
        else:
            self._scripted_committed[env_idx] = False
        tcp_z = cast("Panda", self.agent).tcp.pose.p[:, 2]
        if env_idx is None or not hasattr(self, "_scripted_seat_z_set"):
            self._scripted_seat_z_set = tcp_z.clone()
        else:
            self._scripted_seat_z_set[env_idx] = tcp_z[env_idx]
        if not hasattr(self, "_scripted_tip_trim"):
            self._scripted_tip_trim = torch.zeros((self.num_envs, 2), device=self.device)
        if env_idx is None:
            self._scripted_tip_trim[:] = 0.0
        else:
            self._scripted_tip_trim[env_idx] = 0.0
        self._latch_yaw_command(env_idx)

    def _latch_yaw_command(self, env_idx: torch.Tensor | None) -> None:
        if not hasattr(self, "_socket_yaw"):
            return
        base_q = _ScriptedInsertFamilyMixin._gripper_down_target_q(self)
        tcp_q = cast("Panda", self.agent).tcp.pose.q
        rel = _yaw_between(_body_y_xy(base_q), _body_y_xy(tcp_q))
        cmd = rel + fold_hex(self._socket_yaw - rel)
        if env_idx is None:
            self._scripted_socket_yaw_cmd[:] = cmd
        else:
            self._scripted_socket_yaw_cmd[env_idx] = cmd[env_idx]

    _scripted_seat_z_rate: float = 0.003

    _scripted_auto_comove_headroom: float = 4.0

    _scripted_centering_gain: float = 1.0

    _scripted_tip_trim_gain: float = 0.8
    _scripted_tip_trim_max: float = 0.006

    def _intercept_state(self) -> _InterceptState:
        state = super()._intercept_state()
        tip_world = self.bit.pose * self._bit_tip_offset
        off = tip_world.p[:, :2] - self.plate.pose.p[:, :2]
        dist = torch.linalg.norm(off, dim=-1, keepdim=True)
        weight = (1.0 - dist / self._mouth_half_af).clamp(0.0, 1.0)
        target = state.target_world.clone()
        target[:, :2] = target[:, :2] - self._scripted_centering_gain * weight * off
        trim = self._scripted_tip_trim - self._scripted_tip_trim_gain * off
        trim = trim.clamp(-self._scripted_tip_trim_max, self._scripted_tip_trim_max)
        self._scripted_tip_trim = torch.where(self._scripted_committed.unsqueeze(-1), trim, self._scripted_tip_trim)
        target[:, :2] = target[:, :2] + self._scripted_tip_trim
        over_mouth = dist.squeeze(-1) < self._mouth_half_af
        tcp_z = state.tcp_pos[:, 2]
        lower_z = torch.as_tensor(state.lower_z, dtype=tcp_z.dtype, device=tcp_z.device)
        z_cap = torch.maximum(self._scripted_seat_z_set - self._scripted_seat_z_rate, lower_z)
        z_out = torch.maximum(target[:, 2], z_cap)
        target[:, 2] = torch.where(over_mouth, z_out, target[:, 2])
        self._scripted_seat_z_set = torch.where(over_mouth, z_out, tcp_z)
        return state._replace(target_world=target)

    def _scripted_commit_window(self) -> tuple[torch.Tensor, torch.Tensor]:
        r_in, r_out = super()._scripted_commit_window()
        tip_world = self.bit.pose * self._bit_tip_offset
        rel = (self.plate.pose.inv() * tip_world).p
        tip_lateral = torch.linalg.norm(rel[:, :2], dim=-1)

        self._scripted_committed = self._scripted_committed | (
            (rel[:, 2] < self._plate_half_height) & (tip_lateral < self._mouth_half_af)
        )

        tcp_xy = cast("Panda", self.agent).tcp.pose.p[:, :2]
        ee_lateral = torch.linalg.norm(tcp_xy - self.plate.pose.p[:, :2], dim=-1)
        shift = ee_lateral - tip_lateral
        r_in_t = (r_in + shift).clamp(min=0.0)
        r_out_t = torch.maximum(r_out + shift, r_in_t + self._scripted_center_min_window)

        held_in = torch.full_like(tip_lateral, float(r_out))
        r_in_t = torch.where(self._scripted_committed, held_in, r_in_t)
        r_out_t = torch.where(self._scripted_committed, held_in * self._scripted_commit_hold_scale, r_out_t)
        return r_in_t, r_out_t

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._clear_lateral

    @property
    def _scripted_success_clearance(self) -> float | None:
        return self._clear_lateral

    @property
    def _scripted_socket_chamfer(self) -> float:
        return self._chamfer

    @property
    def _tip_below_tcp(self) -> float:
        return self._bit_grasp_offset + self._bit_half_length

    def _scripted_collar_top_z(self, deck_z: float) -> float:
        return deck_z + 2.0 * self._plate_half_height + self._chamfer

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, float]:
        plate_top = deck_z + 2.0 * self._plate_half_height
        tip_below = self._tip_below_tcp
        wait_z = self._scripted_collar_top_z(deck_z) + self._scripted_wait_clearance + tip_below
        lower_z = plate_top - (self._insert_depth + self._scripted_seat_depth_margin) + tip_below
        return wait_z, lower_z

    def _scripted_hold_gripper(self) -> torch.Tensor:
        target = self._bit_half_af - self._scripted_hold_margin
        value = self._gripper_action_for_aperture(2.0 * target)
        return torch.full((self.num_envs, 1), value, device=self.device, dtype=torch.float32)

    def _gripper_down_target_q(self) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

        base_q = super()._gripper_down_target_q()
        if not hasattr(self, "_scripted_socket_yaw_cmd"):
            return base_q
        half = 0.5 * self._scripted_socket_yaw_cmd
        q_yaw = torch.zeros_like(base_q)
        q_yaw[:, 0] = torch.cos(half)
        q_yaw[:, 3] = torch.sin(half)
        return quaternion_multiply(q_yaw, base_q)


def _body_y_xy(q: torch.Tensor) -> torch.Tensor:
    from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

    return quaternion_to_matrix(q)[:, :2, 1]


def _yaw_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / torch.linalg.norm(a, dim=-1, keepdim=True).clamp(min=1e-9)
    b = b / torch.linalg.norm(b, dim=-1, keepdim=True).clamp(min=1e-9)
    return torch.atan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(dim=-1))
