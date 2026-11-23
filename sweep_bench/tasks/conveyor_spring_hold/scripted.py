from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@dataclass(frozen=True)
class _TabGraspPoint:
    pose: Pose
    linear_velocity: torch.Tensor


class ScriptedSpringHoldMixin(_ScriptedGraspFamilyMixin):
    _PHASE_TRACK = 0
    _PHASE_PULL = 1

    carrier: Actor
    tab: Actor
    _tab_rest_dx: float
    _tab_half_x: float
    _target_extension: torch.Tensor

    @property
    def _tab_grasp_z_above(self) -> float:
        raise NotImplementedError

    _scripted_approach_z_above: float = 0.16
    _scripted_xy_align_tol: float = 0.014
    _scripted_settle_max_steps: int = 4
    _scripted_close_xy_tol: float = 0.016
    _scripted_close_z_tol: float = 0.012
    _scripted_close_steps: int = 4

    _pull_rate: float = 0.008
    _scripted_pull_overshoot: float = 0.0
    _pull_comove_gain: float = 2.8

    @property
    def _scripted_rider(self) -> Actor:
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        p = self.tab.pose.p.clone()
        p[:, 2] = deck_z + self._tab_grasp_z_above
        return cast("Actor", _TabGraspPoint(Pose.create_from_pq(p), self.tab.linear_velocity))

    def _stroke_dir_xy(self) -> torch.Tensor:
        rot = quaternion_to_matrix(self.carrier.pose.q)
        u = rot[:, :2, 0]
        return u / torch.linalg.norm(u, dim=-1, keepdim=True).clamp(min=1e-9)

    def _pull_position_input(self, target_world: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
        pos_step = self._auto_scaled(self._scripted_pos_step)
        cap = torch.as_tensor(
            pos_step.unsqueeze(-1) if isinstance(pos_step, torch.Tensor) else pos_step,
            dtype=target_world.dtype,
            device=target_world.device,
        )
        delta = target_world - tcp_pos
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap / xy_norm, max=1.0)
        z = torch.maximum(torch.minimum(delta[:, 2:3], cap), -cap)
        return torch.clip(torch.cat([xy, z], dim=-1) / 0.1, -1.0, 1.0)

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_pull_phase"):
            self._pull_phase = torch.full((b,), self._PHASE_TRACK, dtype=torch.long, device=device)
            self._pull_cmd = torch.zeros((b,), device=device)
            self._pull_z = torch.zeros((b,), device=device)
        if env_idx is None:
            self._pull_phase[:] = self._PHASE_TRACK
            self._pull_cmd[:] = 0.0
        else:
            self._pull_phase[env_idx] = self._PHASE_TRACK
            self._pull_cmd[env_idx] = 0.0

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_pull_phase"):
            self.reset_scripted()

        grasp_action = super().compute_scripted_action()

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q

        pull = self._pull_phase
        closed = self._scripted_phase >= self._PHASE_LIFT
        entering = closed & (pull == self._PHASE_TRACK)
        self._pull_z = torch.where(entering, tcp_pos[:, 2], self._pull_z)
        pull = torch.where(closed, torch.full_like(pull, self._PHASE_PULL), pull)
        self._pull_phase = pull
        in_pull = pull == self._PHASE_PULL

        speed = torch.abs(self.belt.velocity)
        goal = self._target_extension + self._scripted_pull_overshoot
        self._pull_cmd = torch.where(
            in_pull,
            torch.minimum(self._pull_cmd + self._pull_rate, goal),
            torch.zeros_like(self._pull_cmd),
        )

        carrier_pos = self.carrier.pose.p
        u = self._stroke_dir_xy()
        d = torch.zeros_like(u)
        d[:, 1] = self.belt.direction_sign
        lead = self._pull_comove_gain * speed / self.control_freq
        stroke_x = (self._tab_rest_dx - self._pull_cmd).unsqueeze(-1)
        target_xy = carrier_pos[:, :2] + u * stroke_x + d * lead.unsqueeze(-1)

        target = tcp_pos.clone()
        target[:, 0] = target_xy[:, 0]
        target[:, 1] = target_xy[:, 1]
        target[:, 2] = self._pull_z

        pos_input = self._pull_position_input(target, tcp_pos)
        rot_input = self._rotation_input(tcp_q)
        _open_val, close_val = self._scripted_gripper_values()
        gripper = torch.full((self.num_envs, 1), close_val, device=self.device, dtype=torch.float32)
        pull_action = torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)

        return torch.where(in_pull.unsqueeze(-1), pull_action, grasp_action)
