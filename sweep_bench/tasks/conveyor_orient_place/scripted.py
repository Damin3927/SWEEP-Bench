from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedOrientPlaceMixin(_ScriptedPlaceFamilyMixin):
    tray: Actor
    _slot_floor_half_height: float
    _slot_wall_half_height: float
    _block_half_z: float
    _block_half_y: float
    _tray_origin_above_deck: float
    _scripted_slot_yaw: torch.Tensor

    _scripted_accel_step: float | None = 0.0025

    _scripted_rot_gain: float = 0.6

    _scripted_wait_clearance: float = 0.015
    _scripted_place_gap: float = -0.003
    _scripted_release_finger_margin: float = 0.008

    _scripted_center_r_in: float = 0.005
    _scripted_center_r_out: float = 0.012
    _scripted_auto_park_v_lo: float = 0.10
    _scripted_auto_park_v_hi: float = 0.12
    _scripted_lower_xy_tol: float = 0.010

    @property
    def _scripted_rider(self) -> Actor:
        return self.tray

    def _wall_top_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2] + self._slot_floor_half_height + 2.0 * self._slot_wall_half_height

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return self._wall_top_z(deck_z, rider_pos)

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wall_top = self._wall_top_z(deck_z, rider_pos)
        wait_z = wall_top + self._block_half_z + self._scripted_wait_clearance
        lower_z = wall_top + self._block_half_z + self._scripted_place_gap
        return wait_z, lower_z

    def _release_finger_target(self) -> torch.Tensor:
        target = self._block_half_y + self._scripted_release_finger_margin
        return torch.full((self.num_envs,), min(target, 0.04), device=self.device, dtype=torch.float32)

    def _scripted_release_gripper(self) -> torch.Tensor:
        return self._gripper_action_for_aperture(2.0 * self._release_finger_target()).to(torch.float32)

    def _gripper_down_target_q(self) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

        base_q = super()._gripper_down_target_q()
        if not hasattr(self, "_scripted_slot_yaw"):
            return base_q

        def yawed(extra: float) -> torch.Tensor:
            half = 0.5 * (self._scripted_slot_yaw + extra)
            q_yaw = torch.zeros_like(base_q)
            q_yaw[:, 0] = torch.cos(half)
            q_yaw[:, 3] = torch.sin(half)
            return quaternion_multiply(q_yaw, base_q)

        native = yawed(0.0)
        flipped = yawed(math.pi)
        tcp_q = cast("Panda", self.agent).tcp.pose.q
        nearer = (native * tcp_q).sum(-1).abs() >= (flipped * tcp_q).sum(-1).abs()
        return torch.where(nearer.unsqueeze(-1), native, flipped)
