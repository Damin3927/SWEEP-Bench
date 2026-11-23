from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedCartCoupleMixin(_ScriptedGraspFamilyMixin):
    lead: Actor
    follower: Actor
    _mated_center_distance: float
    _stub_half_x: float

    _scripted_approach_z_above: float = 0.16

    _CC_GRASPING = 0
    _CC_ARREST = 1
    _CC_RELEASE = 2
    _CC_RETREAT = 3

    _cc_grasp_check_dist: float = 0.05
    _cc_min_arrest_steps: int = 4
    _cc_couple_gap: float = 0.006
    _cc_release_steps: int = 4
    _cc_retreat_dx: float = -0.10
    _cc_retreat_dz: float = 0.18
    _cc_retreat_clear_dz: float = 0.04
    _cc_grip_squeeze: float = 0.0012

    def _scripted_close_done(self, in_close: torch.Tensor) -> torch.Tensor:
        gripped = self.gripper_aperture() <= 2.0 * self._stub_half_x - self._cc_grip_squeeze
        capped = self._scripted_close_counter >= self._scripted_close_steps
        return in_close & (gripped | capped)

    @property
    def _scripted_rider(self) -> Actor:
        return self.lead

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_cc_phase"):
            self._cc_phase = torch.full((b,), self._CC_GRASPING, dtype=torch.long, device=device)
            self._cc_arrest_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._cc_release_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._cc_hold_pos = torch.zeros((b, 3), device=device)
        if env_idx is None:
            self._cc_phase[:] = self._CC_GRASPING
            self._cc_arrest_counter[:] = 0
            self._cc_release_counter[:] = 0
            self._cc_hold_pos[:] = 0.0
        else:
            self._cc_phase[env_idx] = self._CC_GRASPING
            self._cc_arrest_counter[env_idx] = 0
            self._cc_release_counter[env_idx] = 0
            self._cc_hold_pos[env_idx] = 0.0

    def _cc_arrest_target(self, tcp_pos: torch.Tensor) -> torch.Tensor:
        return self._cc_hold_pos

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_cc_phase"):
            self.reset_scripted()

        grasp_action = super().compute_scripted_action()

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        lead_pos = self.lead.pose.p

        cc = self._cc_phase

        grasp_closed = self._scripted_phase >= self._PHASE_LIFT
        stub_in_hand = torch.linalg.norm(tcp_pos - lead_pos, dim=-1) < self._cc_grasp_check_dist
        to_arrest = (cc == self._CC_GRASPING) & grasp_closed & stub_in_hand
        ox = self.workspace_offset[0]
        recentred = tcp_pos.clone()
        recentred[:, 0] = recentred[:, 0] + (ox - lead_pos[:, 0])
        self._cc_hold_pos = torch.where(to_arrest.unsqueeze(-1), recentred, self._cc_hold_pos)
        cc = torch.where(to_arrest, torch.full_like(cc, self._CC_ARREST), cc)

        sign = self.belt.direction_sign
        follower_pos = self.follower.pose.p
        couple_gap = (lead_pos[:, 1] - follower_pos[:, 1]) * sign - self._mated_center_distance
        in_arrest = cc == self._CC_ARREST
        self._cc_arrest_counter = torch.where(
            in_arrest,
            self._cc_arrest_counter + 1,
            torch.zeros_like(self._cc_arrest_counter),
        )
        to_release = (
            in_arrest & (couple_gap < self._cc_couple_gap) & (self._cc_arrest_counter >= self._cc_min_arrest_steps)
        )
        cc = torch.where(to_release, torch.full_like(cc, self._CC_RELEASE), cc)

        in_release = cc == self._CC_RELEASE
        self._cc_release_counter = torch.where(
            in_release,
            self._cc_release_counter + 1,
            torch.zeros_like(self._cc_release_counter),
        )
        to_retreat = in_release & (self._cc_release_counter >= self._cc_release_steps)
        cc = torch.where(to_retreat, torch.full_like(cc, self._CC_RETREAT), cc)

        self._cc_phase = cc

        hold = self._cc_arrest_target(tcp_pos)
        target = tcp_pos.clone()
        is_hold = (cc == self._CC_ARREST) | (cc == self._CC_RELEASE)
        target = torch.where(is_hold.unsqueeze(-1), hold, target)

        retreat = hold.clone()
        retreat[:, 2] = retreat[:, 2] + self._cc_retreat_dz
        pads_clear = tcp_pos[:, 2] >= hold[:, 2] + self._cc_retreat_clear_dz
        retreat[:, 0] = torch.where(pads_clear, retreat[:, 0] + self._cc_retreat_dx, retreat[:, 0])
        target = torch.where((cc == self._CC_RETREAT).unsqueeze(-1), retreat, target)

        pos_input = self._position_input(target, tcp_pos)
        rot_input = self._rotation_input(tcp_q)

        open_val, close_val = self._scripted_gripper_values()
        is_open = (cc == self._CC_RELEASE) | (cc == self._CC_RETREAT)
        gripper = torch.where(
            is_open,
            torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32),
            torch.full((self.num_envs,), close_val, device=self.device, dtype=torch.float32),
        ).unsqueeze(-1)
        chain_action = torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)

        in_chain = cc != self._CC_GRASPING
        return torch.where(in_chain.unsqueeze(-1), chain_action, grasp_action)
