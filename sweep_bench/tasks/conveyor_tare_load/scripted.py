from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted_families import (
    ScriptedGraspMixin as _ScriptedGraspFamilyMixin,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


class ScriptedTareLoadMixin(_ScriptedGraspFamilyMixin):
    part: Actor
    pan_well: Actor
    plunger: Actor
    plunger_well: Actor
    _part_half_z: torch.Tensor
    _pan_well_origin_above_surface: float
    _pan_rim_top_above_well: float
    _part_min_half_height: float
    _finger_drop_below_tcp: float
    _plunger_face_above_well: float
    _plunger_anchor_local_z: float

    _scripted_lift_z_above: float = 0.16
    _scripted_hold_z_above: float = 0.12
    _scripted_approach_z_above: float = 0.15

    @property
    def _scripted_close_z_tol(self) -> float:
        return self._part_min_half_height

    _scripted_pos_step = 0.014
    _scripted_v_ee_max = 0.21

    _TL_GRASPING = 0
    _TL_SECURE = 1
    _TL_TO_TARE = 2
    _TL_PRESS = 3
    _TL_RISE = 4
    _TL_TO_PAN = 5
    _TL_LOWER = 6
    _TL_RELEASE = 7
    _TL_RETREAT = 8

    _tl_transport_pos_step: float = 0.045
    _tl_transport_accel_step: float = 0.006
    _tl_carry_z_above: float = 0.12
    _tl_station_clearance_above_rim: float = 0.129

    @property
    def _tl_station_z_above_surface(self) -> float:
        return self._pan_well_origin_above_surface + self._pan_rim_top_above_well + self._tl_station_clearance_above_rim

    _tl_secure_settle_speed: float = 0.15
    _tl_secure_max_steps: int = 20
    _tl_grasp_check_dist: float = 0.06
    _tl_belt_clear_dx: float = 0.14
    _tl_station_xy_tol: float = 0.012
    _tl_press_aim_depth: float = 0.0085
    _tl_press_commit_depth: float = 0.006
    _tl_press_dwell: int = 5
    _tl_rise_z_tol: float = 0.01
    _tl_rim_clearance: float = 0.05
    _tl_pan_land_tol: float = 0.010
    _tl_release_rim_gap: float = 0.004
    _tl_pad_rim_clearance: float = 0.004
    _tl_lower_below_release: float = 0.02
    _tl_release_z_tol: float = 0.006
    _scripted_release_steps: int = 10
    _tl_reach_slow_lo: float = 0.66
    _tl_reach_slow_hi: float = 0.74
    _tl_reach_slowdown: float = 0.85
    _tl_approach_reach_lo: float = 0.66
    _tl_approach_reach_hi: float = 0.74
    _tl_approach_bearing_lo: float = 0.35
    _tl_approach_bearing_hi: float = 0.70
    _tl_approach_hold_r_cap: float = 0.62

    @property
    def _scripted_rider(self) -> Actor:
        return self.part

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_tl_phase"):
            self._tl_phase = torch.full((b,), self._TL_GRASPING, dtype=torch.long, device=device)
            self._tl_release_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._tl_secure_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._tl_press_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._tl_secure_xy = torch.zeros((b, 2), device=device)
            self._tl_press_grip_drop = torch.zeros((b,), device=device)
            self._tl_release_z = torch.zeros((b,), device=device)
            self._tl_prev_delta = torch.zeros((b, 3), device=device)
        if env_idx is None:
            self._tl_phase[:] = self._TL_GRASPING
            self._tl_release_counter[:] = 0
            self._tl_secure_counter[:] = 0
            self._tl_press_counter[:] = 0
            self._tl_press_grip_drop[:] = 0.0
            self._tl_release_z[:] = 0.0
            self._tl_prev_delta[:] = 0.0
        else:
            self._tl_phase[env_idx] = self._TL_GRASPING
            self._tl_release_counter[env_idx] = 0
            self._tl_secure_counter[env_idx] = 0
            self._tl_press_counter[env_idx] = 0
            self._tl_press_grip_drop[env_idx] = 0.0
            self._tl_release_z[env_idx] = 0.0
            self._tl_prev_delta[env_idx] = 0.0

    def _transport_position_input(
        self, target_world: torch.Tensor, tcp_pos: torch.Tensor, ramping: torch.Tensor
    ) -> torch.Tensor:
        delta = target_world - tcp_pos
        accel = self._tl_transport_accel_step
        prev = self._tl_prev_delta
        delta = prev + (delta - prev).clamp(-accel, accel)
        base_xy = self.agent.robot.pose.p[:, :2]
        r_tcp = torch.linalg.norm(tcp_pos[:, :2] - base_xy, dim=-1)
        near_reach = ((r_tcp - self._tl_reach_slow_lo) / (self._tl_reach_slow_hi - self._tl_reach_slow_lo)).clamp(
            0.0, 1.0
        )
        cap = (self._tl_transport_pos_step * (1.0 - self._tl_reach_slowdown * near_reach)).unsqueeze(-1)
        xy = delta[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(cap / xy_norm, max=1.0)
        z = torch.maximum(torch.minimum(delta[:, 2:3], cap), -cap)
        clamped = torch.cat([xy, z], dim=-1)
        self._tl_prev_delta = torch.where(ramping.unsqueeze(-1), clamped, torch.zeros_like(clamped))
        return torch.clip(clamped / 0.1, -1.0, 1.0)

    def _station_approach_xy(self, tcp_xy: torch.Tensor, station_xy: torch.Tensor) -> torch.Tensor:
        base_xy = self.agent.robot.pose.p[:, :2]
        d_grasp = self._tl_secure_xy - base_xy
        d_station = station_xy - base_xy
        d_tcp = tcp_xy - base_xy
        th_grasp = torch.atan2(d_grasp[:, 1], d_grasp[:, 0])
        th_station = torch.atan2(d_station[:, 1], d_station[:, 0])
        th_tcp = torch.atan2(d_tcp[:, 1], d_tcp[:, 0])
        r_station = torch.linalg.norm(d_station, dim=-1)
        r_grasp = torch.linalg.norm(d_grasp, dim=-1)
        th_span = th_grasp - th_station
        frac = torch.where(
            th_span.abs() > 1e-3,
            ((th_tcp - th_station) / th_span).clamp(0.0, 1.0),
            torch.zeros_like(th_span),
        )
        radius_reach = (
            (r_grasp - self._tl_approach_reach_lo) / (self._tl_approach_reach_hi - self._tl_approach_reach_lo)
        ).clamp(0.0, 1.0)
        bearing_reach = (
            (th_span.abs() - self._tl_approach_bearing_lo)
            / (self._tl_approach_bearing_hi - self._tl_approach_bearing_lo)
        ).clamp(0.0, 1.0)
        reach = torch.maximum(radius_reach, bearing_reach)
        comfortable = r_grasp < self._tl_approach_reach_lo
        r_hold = torch.where(comfortable, r_grasp, torch.clamp(r_grasp, max=self._tl_approach_hold_r_cap))
        r_target = r_station + (r_hold - r_station) * frac * reach
        heading = torch.stack([torch.cos(th_station), torch.sin(th_station)], dim=-1)
        return base_xy + r_target.unsqueeze(-1) * heading

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_tl_phase"):
            self.reset_scripted()

        grasp_action = super().compute_scripted_action()

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        part_pos = self.part.pose.p
        pan_well_pos = self.pan_well.pose.p
        plunger_well_pos = self.plunger_well.pose.p

        ox, _oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        carry_z = deck_z + self._tl_carry_z_above
        station_z = oz + self._tl_station_z_above_surface
        half_z = self._part_half_z
        rim_top = pan_well_pos[:, 2] + self._pan_rim_top_above_well
        plunger_face_z = plunger_well_pos[:, 2] + self._plunger_face_above_well
        press_depth = self._plunger_anchor_local_z - (self.plunger.pose.p[:, 2] - plunger_well_pos[:, 2])
        grip_drop = tcp_pos[:, 2] - (part_pos[:, 2] - half_z)

        tl = self._tl_phase

        agent_speed = torch.linalg.norm(agent.tcp.linear_velocity, dim=-1)

        grasp_done = self._scripted_phase == self._PHASE_HOLD
        part_in_hand = torch.linalg.norm(tcp_pos - part_pos, dim=-1) < self._tl_grasp_check_dist
        to_secure = (tl == self._TL_GRASPING) & grasp_done & part_in_hand
        self._tl_secure_xy = torch.where(to_secure.unsqueeze(-1), tcp_pos[:, :2], self._tl_secure_xy)
        tl = torch.where(to_secure, torch.full_like(tl, self._TL_SECURE), tl)

        in_secure = tl == self._TL_SECURE
        self._tl_secure_counter = torch.where(
            in_secure, self._tl_secure_counter + 1, torch.zeros_like(self._tl_secure_counter)
        )
        settled_arm = (agent_speed < self._tl_secure_settle_speed) | (
            self._tl_secure_counter >= self._tl_secure_max_steps
        )
        tl = torch.where(in_secure & settled_arm, torch.full_like(tl, self._TL_TO_TARE), tl)

        over_plunger = torch.linalg.norm(tcp_pos[:, :2] - plunger_well_pos[:, :2], dim=-1) < self._tl_station_xy_tol
        at_station_z = tcp_pos[:, 2] < station_z + self._tl_rise_z_tol
        to_press = (tl == self._TL_TO_TARE) & over_plunger & at_station_z
        self._tl_press_grip_drop = torch.where(to_press, grip_drop, self._tl_press_grip_drop)
        tl = torch.where(to_press, torch.full_like(tl, self._TL_PRESS), tl)

        in_press = tl == self._TL_PRESS
        pressing = in_press & (press_depth >= self._tl_press_commit_depth)
        self._tl_press_counter = torch.where(
            pressing, self._tl_press_counter + 1, torch.zeros_like(self._tl_press_counter)
        )
        tl = torch.where(
            in_press & (self._tl_press_counter >= self._tl_press_dwell), torch.full_like(tl, self._TL_RISE), tl
        )

        tl = torch.where(
            (tl == self._TL_RISE) & (tcp_pos[:, 2] > station_z - self._tl_rise_z_tol),
            torch.full_like(tl, self._TL_TO_PAN),
            tl,
        )

        over_pan = torch.linalg.norm(tcp_pos[:, :2] - pan_well_pos[:, :2], dim=-1) < self._tl_station_xy_tol
        tl = torch.where((tl == self._TL_TO_PAN) & over_pan, torch.full_like(tl, self._TL_LOWER), tl)

        live_release_z = torch.maximum(
            rim_top + self._tl_release_rim_gap + grip_drop,
            rim_top + self._tl_pad_rim_clearance + self._finger_drop_below_tcp,
        )
        to_release = (tl == self._TL_LOWER) & (tcp_pos[:, 2] <= live_release_z + self._tl_release_z_tol)
        self._tl_release_z = torch.where(to_release, live_release_z, self._tl_release_z)
        tl = torch.where(to_release, torch.full_like(tl, self._TL_RELEASE), tl)

        in_release = tl == self._TL_RELEASE
        self._tl_release_counter = torch.where(
            in_release, self._tl_release_counter + 1, torch.zeros_like(self._tl_release_counter)
        )
        tl = torch.where(
            in_release & (self._tl_release_counter >= self._scripted_release_steps),
            torch.full_like(tl, self._TL_RETREAT),
            tl,
        )

        self._tl_phase = tl

        target = tcp_pos.clone()

        is_secure = tl == self._TL_SECURE
        target[:, 0] = torch.where(is_secure, self._tl_secure_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_secure, self._tl_secure_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_secure, torch.full_like(target[:, 2], carry_z), target[:, 2])

        approach_xy = self._station_approach_xy(tcp_pos[:, :2], plunger_well_pos[:, :2])
        belt_cleared = tcp_pos[:, 0] < ox - self._tl_belt_clear_dx
        travel_z = torch.where(
            belt_cleared, torch.full_like(target[:, 2], station_z), torch.full_like(target[:, 2], carry_z)
        )
        is_to_tare = tl == self._TL_TO_TARE
        target[:, 0] = torch.where(is_to_tare, approach_xy[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_to_tare, approach_xy[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_to_tare, travel_z, target[:, 2])

        is_press = tl == self._TL_PRESS
        press_aim_z = plunger_face_z - self._tl_press_aim_depth + self._tl_press_grip_drop
        press_z = torch.where(press_depth < self._tl_press_commit_depth, press_aim_z, tcp_pos[:, 2])
        target[:, 0] = torch.where(is_press, plunger_well_pos[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_press, plunger_well_pos[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_press, press_z, target[:, 2])

        is_rise = tl == self._TL_RISE
        target[:, 0] = torch.where(is_rise, plunger_well_pos[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_rise, plunger_well_pos[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_rise, torch.full_like(target[:, 2], station_z), target[:, 2])

        is_to_pan = tl == self._TL_TO_PAN
        target[:, 0] = torch.where(is_to_pan, pan_well_pos[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_to_pan, pan_well_pos[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_to_pan, torch.full_like(target[:, 2], station_z), target[:, 2])

        is_lower = tl == self._TL_LOWER
        centred = torch.linalg.norm(part_pos[:, :2] - pan_well_pos[:, :2], dim=-1) < self._tl_pan_land_tol
        descend_z = live_release_z - self._tl_lower_below_release
        rim_hover_z = rim_top + self._tl_rim_clearance + grip_drop
        target[:, 0] = torch.where(is_lower, pan_well_pos[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_lower, pan_well_pos[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_lower, torch.where(centred, descend_z, rim_hover_z), target[:, 2])

        is_release = tl == self._TL_RELEASE
        target[:, 0] = torch.where(is_release, pan_well_pos[:, 0], target[:, 0])
        target[:, 1] = torch.where(is_release, pan_well_pos[:, 1], target[:, 1])
        target[:, 2] = torch.where(is_release, self._tl_release_z, target[:, 2])

        is_retreat = tl == self._TL_RETREAT
        target[:, 2] = torch.where(is_retreat, torch.full_like(target[:, 2], carry_z), target[:, 2])

        in_transport = tl != self._TL_GRASPING
        pos_input = self._transport_position_input(target, tcp_pos, in_transport)
        rot_input = self._rotation_input(tcp_q)

        open_val, close_val = self._scripted_gripper_values()
        is_open = (tl == self._TL_RELEASE) | (tl == self._TL_RETREAT)
        gripper = torch.where(
            is_open,
            torch.full((self.num_envs,), open_val, device=self.device, dtype=torch.float32),
            torch.full((self.num_envs,), close_val, device=self.device, dtype=torch.float32),
        ).unsqueeze(-1)
        transport_action = torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)

        return torch.where(in_transport.unsqueeze(-1), transport_action, grasp_action)
