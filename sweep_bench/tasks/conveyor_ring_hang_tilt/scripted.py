from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class _AimPose:
    __slots__ = ("p",)

    def __init__(self, p: torch.Tensor) -> None:
        self.p = p


class _PostAxisAim:
    def __init__(self, env: ScriptedRingHangTiltMixin) -> None:
        self._env = env

    @property
    def pose(self) -> _AimPose:
        return _AimPose(self._env.post_axis_aim_world())


class _AimBand(NamedTuple):
    lower_z: float
    wait_z: float


class ScriptedRingHangTiltMixin(_ScriptedPlaceFamilyMixin):
    stand: Actor
    _base_half_height: float
    _post_height: float
    _post_radius: float
    _ring_half_height: float
    _ring_half_width: float
    _aperture_half_width: float

    _scripted_accel_step: float | None = 0.005
    _scripted_accel_step_z: float = 0.0025

    _scripted_wait_clearance: float = 0.015
    _scripted_engage: float = 0.008

    _plunge_brake_frac: float = 0.5
    _max_plunge_step: float = 0.006
    _scripted_release_finger_margin: float = 0.014
    _scripted_grip_squeeze: float = 0.001

    _scripted_center_r_in: float = 0.012
    _scripted_center_r_out: float = 0.025
    _scripted_lower_xy_tol: float = 0.015
    _scripted_lower_z_tol: float = 0.012

    def post_axis_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def _post_tilt_rad(self) -> float:
        raise NotImplementedError

    @property
    def _scripted_release_along(self) -> float:
        return self._post_height - self._scripted_engage + self._ring_half_height

    @property
    def _scripted_wait_along(self) -> float:
        return self._post_height + self._scripted_wait_clearance + self._ring_half_height

    def _scripted_aim_band(self) -> _AimBand:
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        cos_tilt = math.cos(self._post_tilt_rad)
        root_z = deck_z + 2.0 * self._base_half_height
        return _AimBand(
            lower_z=root_z + self._scripted_release_along * cos_tilt,
            wait_z=root_z + self._scripted_wait_along * cos_tilt,
        )

    def post_axis_aim_world(self) -> torch.Tensor:
        root, axis = self.post_axis_world()
        band = self._scripted_aim_band()
        tcp_z = cast("Panda", self.agent).tcp.pose.p[:, 2]
        z_ref = tcp_z.clamp(band.lower_z, band.wait_z)
        along_ref = (z_ref - root[:, 2]) / math.cos(self._post_tilt_rad)
        return root + axis * along_ref.unsqueeze(-1)

    @property
    def _scripted_rider(self) -> Actor:
        if not hasattr(self, "_scripted_axis_aim"):
            self._scripted_axis_aim = _PostAxisAim(self)
        return cast("Actor", self._scripted_axis_aim)

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> float:
        return deck_z

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, float]:
        band = self._scripted_aim_band()
        return band.wait_z, band.lower_z

    def _scripted_gripper_values(self) -> tuple[float, float]:
        open_val, _close_val = super()._scripted_gripper_values()
        target = self._ring_half_width - self._scripted_grip_squeeze
        return open_val, self._gripper_action_for_aperture(2.0 * target)

    def _release_finger_target(self) -> float:
        return min(self._ring_half_width + self._scripted_release_finger_margin, 0.04)

    def _scripted_release_gripper(self) -> torch.Tensor:
        value = self._gripper_action_for_aperture(2.0 * self._release_finger_target())
        return torch.full((self.num_envs,), value, device=self.device, dtype=torch.float32)

    def compute_scripted_action(self) -> torch.Tensor:
        if hasattr(self, "_scripted_phase"):
            band = self._scripted_aim_band()
            tcp_pos = cast("Panda", self.agent).tcp.pose.p
            aim_xy = self.post_axis_aim_world()[:, :2]
            xy_dist = torch.linalg.norm(tcp_pos[:, :2] - aim_xy, dim=-1)
            overshot = (
                (self._scripted_phase == self._PHASE_INTERCEPT)
                & (tcp_pos[:, 2] < band.lower_z)
                & (xy_dist < self._scripted_lower_xy_tol)
            )
            self._scripted_phase = torch.where(
                overshot, torch.full_like(self._scripted_phase, self._PHASE_RELEASE), self._scripted_phase
            )
        return super().compute_scripted_action()

    def _clamp_pos_delta(self, delta_pos: torch.Tensor) -> torch.Tensor:
        prev_dz = self._scripted_prev_delta[:, 2].clone()
        delta = super()._clamp_pos_delta(delta_pos)

        accel_z = torch.as_tensor(
            self._auto_scaled(self._scripted_accel_step_z), dtype=delta.dtype, device=delta.device
        ).expand_as(prev_dz)
        dz = torch.minimum(torch.maximum(delta[:, 2], prev_dz - accel_z), prev_dz + accel_z)

        band = self._scripted_aim_band()
        tcp_z = cast("Panda", self.agent).tcp.pose.p[:, 2]
        remaining = (tcp_z - band.lower_z).clamp(min=0.0)
        cap = (self._plunge_brake_frac * remaining).clamp(min=self._max_plunge_step)
        dz = torch.maximum(dz, -cap)
        delta = torch.cat([delta[:, :2], dz.unsqueeze(-1)], dim=-1)
        self._scripted_prev_delta = delta
        return delta

    def _gripper_down_target_q(self) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

        base_q = super()._gripper_down_target_q()
        half = 0.5 * self._post_tilt_rad
        tilt_q = torch.zeros_like(base_q)
        tilt_q[:, 0] = math.cos(half)
        tilt_q[:, 2] = math.sin(half)
        return quaternion_multiply(tilt_q, base_q)
