from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor

_POS_INPUT_SCALE = 0.1


class ScriptedDepthBandMixin(_ScriptedInsertFamilyMixin):
    plunger: Actor
    _plunger_rest_top_above_deck: float
    _plunger_half_width: float
    _band_centre: torch.Tensor
    _band_centre_default: float

    def plunger_depth(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_accel_step: float | None = 0.0025

    _scripted_wait_clearance: float = 0.02

    _scripted_depth_per_z: float = 0.9
    _scripted_press_dip: float = 0.0013
    _scripted_engage_depth: float = 0.0005
    _scripted_depth_gain: float = 1.5
    _scripted_max_dip: float = 0.0045

    _scripted_slow_zone: float = 0.015
    _scripted_dip_rate: float = 0.008

    _scripted_reach_y: float = 0.45

    @property
    def _scripted_rider(self) -> Actor:
        return self.plunger

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._plunger_half_width

    @property
    def _scripted_socket_chamfer(self) -> float:
        return 0.0

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, torch.Tensor]:
        rest_top = deck_z + self._plunger_rest_top_above_deck
        wait_z = rest_top + self._scripted_wait_clearance

        depth = self.plunger_depth()

        band_shift = self._band_centre - self._band_centre_default
        lower_ff = rest_top - self._scripted_press_dip - band_shift

        lower_servo = lower_ff - self._scripted_depth_gain * (self._band_centre - depth) / self._scripted_depth_per_z

        lower_z = torch.where(depth > self._scripted_engage_depth, lower_servo, lower_ff)
        return wait_z, lower_z.clamp(min=rest_top - self._scripted_max_dip, max=wait_z)

    def _scripted_hold_gripper(self) -> torch.Tensor:
        _, close_val = self._scripted_gripper_values()
        return torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)

    def compute_scripted_action(self) -> torch.Tensor:
        action = super().compute_scripted_action()
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        agent = cast("Panda", self.agent)
        near = agent.tcp.pose.p[:, 2] < deck_z + self._plunger_rest_top_above_deck + self._scripted_slow_zone
        limit = self._scripted_dip_rate / _POS_INPUT_SCALE
        action = action.clone()
        action[:, 2] = torch.where(near, action[:, 2].clamp(min=-limit), action[:, 2])

        oy = self.workspace_offset[1]
        downstream = (self.plunger.pose.p[:, 1] - oy) * self.belt.direction_sign
        past_reach = downstream > self._scripted_reach_y
        action[:, 0] = torch.where(past_reach, torch.zeros_like(action[:, 0]), action[:, 0])
        action[:, 1] = torch.where(past_reach, torch.zeros_like(action[:, 1]), action[:, 1])
        action[:, 2] = torch.where(past_reach, torch.ones_like(action[:, 2]), action[:, 2])
        return action
