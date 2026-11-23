from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)
from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedPlaceMixin as _ScriptedPlaceFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedTrayToBinMixin(_ScriptedGraspFamilyMixin, _ScriptedPlaceFamilyMixin):
    tray: Actor
    bin: Actor
    part: Actor
    _part_half_z: torch.Tensor
    _part_half_width: torch.Tensor
    _bin_floor_half_height: float
    _bin_wall_half_height: float

    _ttb_leg2: bool = False

    _scripted_approach_z_above: float = 0.15
    _scripted_lift_z_above: float = 0.16
    _scripted_hold_z_above: float = 0.13
    _scripted_pos_step: float = 0.014
    _scripted_v_ee_max: float = 0.21
    _scripted_settle_max_steps: int = 5
    _scripted_close_steps: int = 10

    _scripted_hover_z_above: float = 0.13
    _scripted_wait_clearance: float = 0.030
    _scripted_release_clearance: float = 0.005
    _scripted_lower_z_tol: float = 0.008
    _scripted_release_finger_margin: float = 0.012
    _scripted_release_steps: int = 4

    _ttb_grasp_check_dist: float = 0.06
    _ttb_handoff_settle_speed: float = 0.15
    _ttb_handoff_max_steps: int = 8

    @property
    def _scripted_rider(self) -> Actor:
        return self.bin if self._ttb_leg2 else self.part

    def _bin_rim_world_z(self, bin_z: torch.Tensor) -> torch.Tensor:
        return bin_z + self._bin_floor_half_height + 2.0 * self._bin_wall_half_height

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return self._bin_rim_world_z(rider_pos[:, 2])

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rim_z = self._bin_rim_world_z(rider_pos[:, 2])
        wait_z = rim_z + self._part_half_z + self._scripted_wait_clearance
        lower_z = rim_z + self._part_half_z + self._scripted_release_clearance
        return wait_z, lower_z

    def _release_finger_target(self) -> torch.Tensor:
        return torch.clamp(self._part_half_width + self._scripted_release_finger_margin, max=0.04)

    def _scripted_release_gripper(self) -> torch.Tensor:
        return self._gripper_action_for_aperture(2.0 * self._release_finger_target()).to(torch.float32)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_ttb_handed_off"):
            self._ttb_handed_off = torch.zeros((b,), dtype=torch.bool, device=device)
            self._ttb_hold_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._ttb_place_phase = torch.full((b,), self._PHASE_INTERCEPT, dtype=torch.long, device=device)
            self._ttb_place_bias = torch.zeros((b,), device=device)
            self._scripted_release_counter = torch.zeros((b,), dtype=torch.long, device=device)
        if env_idx is None:
            self._ttb_handed_off[:] = False
            self._ttb_hold_counter[:] = 0
            self._ttb_place_phase[:] = self._PHASE_INTERCEPT
            self._ttb_place_bias[:] = 0.0
            self._scripted_release_counter[:] = 0
        else:
            self._ttb_handed_off[env_idx] = False
            self._ttb_hold_counter[env_idx] = 0
            self._ttb_place_phase[env_idx] = self._PHASE_INTERCEPT
            self._ttb_place_bias[env_idx] = 0.0
            self._scripted_release_counter[env_idx] = 0

    def _place_leg_action(self) -> torch.Tensor:
        self._ttb_leg2 = True
        grasp_phase = self._scripted_phase
        grasp_bias = self._scripted_auto_lead_bias
        self._scripted_phase = self._ttb_place_phase
        self._scripted_auto_lead_bias = self._ttb_place_bias
        try:
            return _ScriptedPlaceFamilyMixin.compute_scripted_action(self)
        finally:
            self._ttb_place_phase = self._scripted_phase
            self._ttb_place_bias = self._scripted_auto_lead_bias
            self._scripted_phase = grasp_phase
            self._scripted_auto_lead_bias = grasp_bias
            self._ttb_leg2 = False

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_ttb_handed_off"):
            self.reset_scripted()

        self._velrand_held_scale = self._advance_velrand_step() if self.velocity_randomization.enabled else None
        try:
            return self._chained_action()
        finally:
            self._velrand_held_scale = None

    def _chained_action(self) -> torch.Tensor:
        grasp_action = super().compute_scripted_action()

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        part_in_hand = torch.linalg.norm(tcp_pos - self.part.pose.p, dim=-1) < self._ttb_grasp_check_dist
        at_hold = self._scripted_phase == self._PHASE_HOLD
        self._ttb_hold_counter = torch.where(
            at_hold, self._ttb_hold_counter + 1, torch.zeros_like(self._ttb_hold_counter)
        )
        calm = (torch.linalg.norm(agent.tcp.linear_velocity, dim=-1) < self._ttb_handoff_settle_speed) | (
            self._ttb_hold_counter >= self._ttb_handoff_max_steps
        )
        self._ttb_handed_off = self._ttb_handed_off | (at_hold & part_in_hand & calm)

        pinned = ~self._ttb_handed_off
        self._ttb_place_phase = torch.where(
            pinned, torch.full_like(self._ttb_place_phase, self._PHASE_INTERCEPT), self._ttb_place_phase
        )
        self._ttb_place_bias = torch.where(pinned, torch.zeros_like(self._ttb_place_bias), self._ttb_place_bias)

        place_action = self._place_leg_action()
        return torch.where(self._ttb_handed_off.unsqueeze(-1), place_action, grasp_action)
