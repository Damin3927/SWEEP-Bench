from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class _AimPose:
    p: torch.Tensor


class _TargetCellAim:
    def __init__(self, env: ScriptedKitFillMixin) -> None:
        self._env = env

    @property
    def pose(self) -> _AimPose:
        return _AimPose(self._env.target_cell_world_pos())


class ScriptedKitFillMixin(_ScriptedGraspFamilyMixin, _ScriptedPlaceFamilyMixin):
    part: Actor
    _part_half_z: torch.Tensor
    _cell_mouth_local_z: float
    _part_in_holder: bool

    def target_cell_world_pos(self) -> torch.Tensor:
        raise NotImplementedError

    _kf_leg2: bool = False

    _scripted_center_r_in: float = 0.008
    _scripted_center_r_out: float = 0.05
    _scripted_lower_xy_tol: float = 0.008
    _scripted_lower_z_tol: float = 0.010

    _scripted_wait_clearance: float = 0.035
    _scripted_release_rim_gap: float = 0.002

    _scripted_approach_z_above: float = 0.15
    _scripted_lift_z_above: float = 0.12
    _scripted_hold_z_above: float = 0.09
    _scripted_close_lead_time: float = 0.0
    _scripted_settle_max_steps: int = 5
    _scripted_close_steps: int = 10

    _kf_feed_bias_max: float = 0.10

    _kf_grasp_check_dist: float = 0.06
    _kf_handoff_settle_speed: float = 0.25
    _kf_handoff_max_steps: int = 4

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        holder = self._part_in_holder

        self._scripted_accel_step: float | None = None if holder else 0.0025

        self._scripted_pos_step: float = 0.014 if holder else 0.01

        self._scripted_v_ee_max: float = 0.21 if holder else 0.15

        self._scripted_v_align: float = 0.25 if holder else 0.10

    @property
    def _scripted_hover_z_above(self) -> float:
        return 0.09 if self._part_in_holder else 0.20

    @property
    def _scripted_release_steps(self) -> int:
        return 4 if self._part_in_holder else 5

    @property
    def _scripted_auto_bias_max(self) -> float:
        return 0.0 if (self._part_in_holder and not self._kf_leg2) else self._kf_feed_bias_max

    @property
    def _scripted_rider(self) -> Actor:
        if self._part_in_holder and not self._kf_leg2:
            return self.part
        if not hasattr(self, "_scripted_cell_aim"):
            self._scripted_cell_aim = _TargetCellAim(self)
        return cast("Actor", self._scripted_cell_aim)

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2] + self._cell_mouth_local_z

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mouth_z = rider_pos[:, 2] + self._cell_mouth_local_z
        wait_z = mouth_z + self._scripted_wait_clearance + self._part_half_z
        lower_z = mouth_z + self._scripted_release_rim_gap + self._part_half_z
        return wait_z, lower_z

    def _scripted_release_gripper(self) -> torch.Tensor:
        open_val, _ = self._scripted_gripper_values()
        return torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        if not self._part_in_holder:
            _ScriptedPlaceFamilyMixin.reset_scripted(self, env_idx)
            return

        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_kf_handed_off"):
            self._kf_handed_off = torch.zeros((b,), dtype=torch.bool, device=device)
            self._kf_hold_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._kf_feed_phase = torch.full((b,), self._PHASE_INTERCEPT, dtype=torch.long, device=device)
            self._kf_feed_bias = torch.zeros((b,), device=device)
            self._scripted_release_counter = torch.zeros((b,), dtype=torch.long, device=device)
        if env_idx is None:
            self._kf_handed_off[:] = False
            self._kf_hold_counter[:] = 0
            self._kf_feed_phase[:] = self._PHASE_INTERCEPT
            self._kf_feed_bias[:] = 0.0
            self._scripted_release_counter[:] = 0
        else:
            self._kf_handed_off[env_idx] = False
            self._kf_hold_counter[env_idx] = 0
            self._kf_feed_phase[env_idx] = self._PHASE_INTERCEPT
            self._kf_feed_bias[env_idx] = 0.0
            self._scripted_release_counter[env_idx] = 0

    def _feed_leg_action(self) -> torch.Tensor:
        self._kf_leg2 = True
        grasp_phase = self._scripted_phase
        grasp_bias = self._scripted_auto_lead_bias
        self._scripted_phase = self._kf_feed_phase
        self._scripted_auto_lead_bias = self._kf_feed_bias
        try:
            return _ScriptedPlaceFamilyMixin.compute_scripted_action(self)
        finally:
            self._kf_feed_phase = self._scripted_phase
            self._kf_feed_bias = self._scripted_auto_lead_bias
            self._scripted_phase = grasp_phase
            self._scripted_auto_lead_bias = grasp_bias
            self._kf_leg2 = False

    def compute_scripted_action(self) -> torch.Tensor:
        if not self._part_in_holder:
            return _ScriptedPlaceFamilyMixin.compute_scripted_action(self)

        self._require_ee_delta_pose()
        if not hasattr(self, "_kf_handed_off"):
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
        part_in_hand = torch.linalg.norm(tcp_pos - self.part.pose.p, dim=-1) < self._kf_grasp_check_dist
        at_lift = (self._scripted_phase == self._PHASE_LIFT) | (self._scripted_phase == self._PHASE_HOLD)
        self._kf_hold_counter = torch.where(at_lift, self._kf_hold_counter + 1, torch.zeros_like(self._kf_hold_counter))
        calm = (torch.linalg.norm(agent.tcp.linear_velocity, dim=-1) < self._kf_handoff_settle_speed) | (
            self._kf_hold_counter >= self._kf_handoff_max_steps
        )
        self._kf_handed_off = self._kf_handed_off | (at_lift & part_in_hand & calm)

        pinned = ~self._kf_handed_off
        self._kf_feed_phase = torch.where(
            pinned, torch.full_like(self._kf_feed_phase, self._PHASE_INTERCEPT), self._kf_feed_phase
        )
        self._kf_feed_bias = torch.where(pinned, torch.zeros_like(self._kf_feed_bias), self._kf_feed_bias)

        feed_action = self._feed_leg_action()
        return torch.where(self._kf_handed_off.unsqueeze(-1), feed_action, grasp_action)
