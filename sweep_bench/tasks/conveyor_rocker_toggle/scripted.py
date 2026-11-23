from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedRockerToggleMixin(_ScriptedInsertFamilyMixin):
    rocker: Actor
    _rocker_start_sign: torch.Tensor
    _housing_origin_above_deck: float
    _rocker_max_top_local_z: float

    def _rocker_tilt(self) -> torch.Tensor:
        raise NotImplementedError

    def _rocker_pad_top_local_z(self, pad_raised: bool) -> float:
        raise NotImplementedError

    _scripted_accel_step: float | None = 0.0025

    _scripted_finger_reach: float = 0.0095
    _scripted_wait_clearance: float = 0.010
    _scripted_press_overshoot: float = 0.006

    _scripted_pad_commit_half_width: float = 0.014

    _scripted_withdraw_z_above: float = 0.20
    _scripted_toggle_commit_tilt: float = math.radians(8.0)

    @property
    def _scripted_rider(self) -> Actor:
        return self.rocker

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._scripted_pad_commit_half_width

    @property
    def _scripted_socket_chamfer(self) -> float:
        return 0.006

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, float]:
        base_z = deck_z + self._housing_origin_above_deck
        wait_z = base_z + self._rocker_max_top_local_z + self._scripted_finger_reach + self._scripted_wait_clearance
        lower_z = (
            base_z
            + self._rocker_pad_top_local_z(pad_raised=False)
            + self._scripted_finger_reach
            - self._scripted_press_overshoot
        )
        return wait_z, lower_z

    def _scripted_hold_gripper(self) -> torch.Tensor:
        _, close_val = self._scripted_gripper_values()
        return torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        if not hasattr(self, "_scripted_withdrawn"):
            self._scripted_withdrawn = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        super().reset_scripted(env_idx)
        if env_idx is None:
            self._scripted_withdrawn[:] = False
        else:
            self._scripted_withdrawn[env_idx] = False

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_scripted_withdrawn"):
            self.reset_scripted()

        state = self._intercept_state()
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        target_world = state.target_world.clone()

        flip = -self._rocker_start_sign * self._rocker_tilt()
        self._scripted_withdrawn = self._scripted_withdrawn | (flip >= self._scripted_toggle_commit_tilt)
        withdrawn = self._scripted_withdrawn
        target_world[:, 0] = torch.where(withdrawn, state.tcp_pos[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(withdrawn, state.tcp_pos[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(
            withdrawn,
            torch.full_like(target_world[:, 2], deck_z + self._scripted_withdraw_z_above),
            target_world[:, 2],
        )

        pos_input = self._position_input(target_world, state.tcp_pos)
        rot_input = self._rotation_input(state.tcp_q)
        gripper = self._scripted_hold_gripper()

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
