from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


class ScriptedStampSealMixin(_ScriptedInsertFamilyMixin):
    pad: Actor
    _pad_origin_above_deck: float
    _collar_height: float
    _plunger_pressed_top_local_z: float
    _stamp_face_below_tcp: float
    _stamp_clear_half_width: float
    _stamp_shaft_radius: float

    def _plunger_depth(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_accel_step: float | None = 0.0025

    _scripted_wait_clearance: float = 0.035
    _scripted_press_overshoot: float = 0.012

    _scripted_bore_ramp: float = 0.006

    _scripted_liftoff_z_above: float = 0.22
    _scripted_latch_commit_depth: float = 0.007

    _scripted_hold_margin: float = 0.006

    @property
    def _scripted_rider(self) -> Actor:
        return self.pad

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._stamp_clear_half_width

    @property
    def _scripted_socket_chamfer(self) -> float:
        return self._scripted_bore_ramp

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, float]:
        base_z = deck_z + self._pad_origin_above_deck
        wait_z = base_z + self._collar_height + self._scripted_wait_clearance + self._stamp_face_below_tcp
        lower_z = (
            base_z + self._plunger_pressed_top_local_z - self._scripted_press_overshoot + self._stamp_face_below_tcp
        )
        return wait_z, lower_z

    def _scripted_hold_gripper(self) -> torch.Tensor:
        target = self._stamp_shaft_radius - self._scripted_hold_margin
        value = self._gripper_action_for_aperture(2.0 * target)
        return torch.full((self.num_envs, 1), value, device=self.device, dtype=torch.float32)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        if not hasattr(self, "_scripted_lifted_off"):
            self._scripted_lifted_off = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        super().reset_scripted(env_idx)
        if env_idx is None:
            self._scripted_lifted_off[:] = False
        else:
            self._scripted_lifted_off[env_idx] = False

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_scripted_lifted_off"):
            self.reset_scripted()

        state = self._intercept_state()
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        target_world = state.target_world.clone()

        self._scripted_lifted_off = self._scripted_lifted_off | (
            self._plunger_depth() >= self._scripted_latch_commit_depth
        )
        lifted = self._scripted_lifted_off
        target_world[:, 0] = torch.where(lifted, state.tcp_pos[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(lifted, state.tcp_pos[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(
            lifted,
            torch.full_like(target_world[:, 2], deck_z + self._scripted_liftoff_z_above),
            target_world[:, 2],
        )

        pos_input = self._position_input(target_world, state.tcp_pos)
        rot_input = self._rotation_input(state.tcp_q)
        gripper = self._scripted_hold_gripper()

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
