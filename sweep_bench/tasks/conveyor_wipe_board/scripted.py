from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedInsertMixin as _ScriptedInsertFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedWipeBoardMixin(_ScriptedInsertFamilyMixin):
    board: Actor
    _board_top_above_deck: float
    _pad_face_below_tcp: float
    _face_half_across: float
    _face_half_thickness: float
    _stem_half_grip: float

    def _markers_cleared(self) -> torch.Tensor:
        raise NotImplementedError

    def _pad_face_board_xy(self) -> torch.Tensor:
        raise NotImplementedError

    def _pad_face_world(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def board_top_z(self) -> torch.Tensor:
        raise NotImplementedError

    _scripted_accel_step: float | None = 0.004

    _scripted_wait_clearance: float = 0.012
    _scripted_press_depth: float = 0.005
    _scripted_touch_band: float = 0.030
    _scripted_touch_rate: float = 0.008
    _scripted_max_slide_comp: float = 0.012

    _scripted_scrub_turn: float = 0.010
    _scripted_scrub_lead: float = 0.024
    _scripted_along_gain: float = 0.1
    _scripted_along_max: float = 0.08
    _scripted_along_p_gain: float = 2.5
    _scripted_scrub_world_lead: float = 1.0
    _scripted_scrub_lead_time: float = 0.13
    _scripted_track_limit: float = 0.50
    _scripted_press_tol: float = 0.010
    _scripted_scrub_xy_tol: float = 0.050

    _scripted_scrub_scale_floor: float = 4.0

    _scripted_board_ramp: float = 0.006

    _scripted_pad_board_align: float = 1.0

    _scripted_liftoff_z_above: float = 0.22

    _scripted_hold_margin: float = 0.020

    @property
    def _scripted_rider(self) -> Actor:
        return self.board

    @property
    def _scripted_socket_half_width(self) -> float:
        return self._face_half_across

    @property
    def _scripted_socket_chamfer(self) -> float:
        return self._scripted_board_ramp

    def _tool_slide(self) -> torch.Tensor:
        face_bottom_z = self._pad_face_world()[:, 2] - self._face_half_thickness
        tcp_z = cast("Panda", self.agent).tcp.pose.p[:, 2]
        return (self._pad_face_below_tcp - (tcp_z - face_bottom_z)).clamp(0.0, self._scripted_max_slide_comp)

    def _scripted_wait_lower_z(self, deck_z: float, rider_pos: torch.Tensor) -> tuple[float, torch.Tensor]:
        top_z = deck_z + self._board_top_above_deck
        wait_z = top_z + self._scripted_wait_clearance + self._pad_face_below_tcp
        lower_z = top_z - self._scripted_press_depth + self._pad_face_below_tcp - self._tool_slide()
        return wait_z, lower_z

    def _scripted_hold_gripper(self) -> torch.Tensor:
        target = self._stem_half_grip - self._scripted_hold_margin
        value = self._gripper_action_for_aperture(2.0 * target)
        return torch.full((self.num_envs, 1), value, device=self.device, dtype=torch.float32)

    def _gripper_down_target_q(self) -> torch.Tensor:
        weight = self._scripted_pad_board_align
        if weight == 0.0:
            return super()._gripper_down_target_q()
        saved = self._scripted_motion_dir
        along = quaternion_to_matrix(self.board.pose.q)[:, :2, 1]
        along = along * torch.sign((along * saved).sum(dim=-1, keepdim=True))
        blended = saved * (1.0 - weight) + along * weight
        self._scripted_motion_dir = blended / blended.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        try:
            return super()._gripper_down_target_q()
        finally:
            self._scripted_motion_dir = saved

    def _scrub_target_xy(self, just_engaged: torch.Tensor) -> torch.Tensor:
        local = self._pad_face_board_xy()
        turn = self._scripted_scrub_turn
        direction = torch.where(
            local[:, 0] >= turn,
            -torch.ones_like(self._scripted_scrub_dir),
            torch.where(local[:, 0] <= -turn, torch.ones_like(self._scripted_scrub_dir), self._scripted_scrub_dir),
        )
        self._scripted_scrub_dir = direction

        board_rot = quaternion_to_matrix(self.board.pose.q)
        across_axis = board_rot[:, :2, 0]
        along_axis = board_rot[:, :2, 1]

        travel = torch.zeros_like(along_axis)
        travel[:, 1] = self.belt.direction_sign * torch.abs(self.belt.velocity)
        feed = (travel * along_axis).sum(dim=-1) * self._scripted_scrub_lead_time
        world_feed = travel * (self._scripted_scrub_lead_time * self._scripted_scrub_world_lead)
        feed = feed * (1.0 - self._scripted_scrub_world_lead)
        bias = torch.where(
            just_engaged,
            torch.zeros_like(feed),
            (self._scripted_along_bias - self._scripted_along_gain * local[:, 1]).clamp(
                -self._scripted_along_max, self._scripted_along_max
            ),
        )
        self._scripted_along_bias = bias
        proportional = -self._scripted_along_p_gain * local[:, 1]

        offset = (
            across_axis * (direction * self._scripted_scrub_lead).unsqueeze(-1)
            + along_axis * (feed + bias + proportional).unsqueeze(-1)
            + world_feed
        )
        return self.board.pose.p[:, :2] + offset

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        if not hasattr(self, "_scripted_lifted_off"):
            self._scripted_lifted_off = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._scripted_scrubbing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._scripted_scrub_dir = torch.ones(self.num_envs, device=self.device)
            self._scripted_along_bias = torch.zeros(self.num_envs, device=self.device)
        super().reset_scripted(env_idx)
        if env_idx is None:
            self._scripted_lifted_off[:] = False
            self._scripted_scrubbing[:] = False
            self._scripted_scrub_dir[:] = 1.0
            self._scripted_along_bias[:] = 0.0
        else:
            self._scripted_lifted_off[env_idx] = False
            self._scripted_scrubbing[env_idx] = False
            self._scripted_scrub_dir[env_idx] = 1.0
            self._scripted_along_bias[env_idx] = 0.0

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_scripted_lifted_off"):
            self.reset_scripted()

        state = self._intercept_state()
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        _wait_z, lower_z = self._scripted_wait_lower_z(deck_z, self.board.pose.p)
        target_world = state.target_world.clone()

        engaged = (torch.abs(state.tcp_pos[:, 2] - lower_z) < self._scripted_press_tol) & (
            state.xy_dist < self._scripted_scrub_xy_tol
        )
        just_engaged = engaged & ~self._scripted_scrubbing
        self._scripted_scrubbing = self._scripted_scrubbing | engaged
        scrubbing = self._scripted_scrubbing
        if self._scripted_auto_scale is not None:
            self._scripted_auto_scale = torch.where(
                scrubbing,
                self._scripted_auto_scale.clamp(min=self._scripted_scrub_scale_floor),
                self._scripted_auto_scale,
            )
        scrub_xy = self._scrub_target_xy(just_engaged)
        target_world[:, 0] = torch.where(scrubbing, scrub_xy[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(scrubbing, scrub_xy[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(scrubbing, lower_z, target_world[:, 2])

        oy = self.workspace_offset[1]
        board_run = (self.board.pose.p[:, 1] - oy) * self.belt.direction_sign
        self._scripted_lifted_off = (
            self._scripted_lifted_off | self._markers_cleared() | (board_run > self._scripted_track_limit)
        )
        lifted = self._scripted_lifted_off
        target_world[:, 0] = torch.where(lifted, state.tcp_pos[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(lifted, state.tcp_pos[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(
            lifted,
            torch.full_like(target_world[:, 2], deck_z + self._scripted_liftoff_z_above),
            target_world[:, 2],
        )

        face_bottom = self._pad_face_world()[:, 2] - self._face_half_thickness
        near_surface = (face_bottom - self.board_top_z) < self._scripted_touch_band
        z_floor = state.tcp_pos[:, 2] - self._scripted_touch_rate
        target_world[:, 2] = torch.where(near_surface, torch.maximum(target_world[:, 2], z_floor), target_world[:, 2])

        pos_input = self._position_input(target_world, state.tcp_pos)
        rot_input = self._rotation_input(state.tcp_q)
        gripper = self._scripted_hold_gripper()

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
