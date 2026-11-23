from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import torch

from sweep_bench.tasks.conveyor_base.scripted import ScriptedControlMixin

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    t = x.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class ScriptedGraspMixin(ScriptedControlMixin):
    _PHASE_APPROACH = 0
    _PHASE_DESCEND = 1
    _PHASE_CLOSE = 2
    _PHASE_LIFT = 3
    _PHASE_HOLD = 4

    _scripted_approach_z_above: float = 0.20
    _scripted_lift_z_above: float = 0.28
    _scripted_hold_z_above: float = 0.25
    _scripted_xy_align_tol: float = 0.02
    _scripted_close_xy_tol: float = 0.025
    _scripted_close_z_tol: float = 0.025
    _scripted_close_steps: int = 14
    _scripted_stopped_close_z_tol: float = 0.012
    _scripted_stopped_stall_vz: float = 0.005
    _scripted_close_lead_time: float = 0.08
    _scripted_settle_cos: float = 0.996
    _scripted_settle_max_steps: int = 8

    _scripted_auto_bias_gain: float = 0.3
    _scripted_auto_bias_max: float = 0.10
    _scripted_auto_bias_radius: float = 0.10
    _scripted_auto_bias_cross: float = 0.06
    _scripted_auto_bias_v_lo: float = 0.15
    _scripted_auto_bias_v_hi: float = 0.30

    _scripted_grasp_endgame_scale_floor: float = 3.5

    @property
    def _scripted_rider(self) -> Actor:
        raise NotImplementedError

    def _scripted_close_done(self, in_close: torch.Tensor) -> torch.Tensor:
        return in_close & (self._scripted_close_counter >= self._scripted_close_steps)

    def _scripted_grasp_z(self, rider_pos: torch.Tensor) -> torch.Tensor:
        return rider_pos[:, 2]

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_scripted_phase"):
            self._scripted_phase = torch.full((b,), self._PHASE_APPROACH, dtype=torch.long, device=device)
            self._scripted_close_counter = torch.zeros((b,), dtype=torch.long, device=device)
            self._scripted_settle_counter = torch.zeros((b,), dtype=torch.long, device=device)
        self._reset_scripted_shared(env_idx)

        if env_idx is None:
            self._scripted_phase[:] = self._PHASE_APPROACH
            self._scripted_close_counter[:] = 0
            self._scripted_settle_counter[:] = 0
        else:
            self._scripted_phase[env_idx] = self._PHASE_APPROACH
            self._scripted_close_counter[env_idx] = 0
            self._scripted_settle_counter[env_idx] = 0

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_scripted_phase"):
            self.reset_scripted()

        device = self.device
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        rider = self._scripted_rider
        rider_pos = rider.pose.p
        self._update_auto_speed_scale(tcp_pos, rider_pos)

        phase = self._scripted_phase

        xy_dist = torch.linalg.norm(tcp_pos[:, :2] - rider_pos[:, :2], dim=-1)
        grasp_z = self._scripted_grasp_z(rider_pos)
        z_diff = torch.abs(tcp_pos[:, 2] - grasp_z)

        target_q = self._gripper_down_target_q()
        rot_cos = torch.abs((tcp_q * target_q).sum(dim=-1))
        hover_aligned = (phase == self._PHASE_APPROACH) & (xy_dist < self._scripted_xy_align_tol)
        self._scripted_settle_counter = torch.where(
            hover_aligned,
            self._scripted_settle_counter + 1,
            torch.zeros_like(self._scripted_settle_counter),
        )
        settled = (rot_cos > self._scripted_settle_cos) | (
            self._scripted_settle_counter >= self._scripted_settle_max_steps
        )
        to_descend = hover_aligned & settled
        phase = torch.where(to_descend, torch.full_like(phase, self._PHASE_DESCEND), phase)

        deep_enough = (z_diff < self._scripted_stopped_close_z_tol) | (
            agent.tcp.linear_velocity[:, 2] > -self._scripted_stopped_stall_vz
        )
        to_close = (
            (phase == self._PHASE_DESCEND)
            & (xy_dist < self._scripted_close_xy_tol)
            & (z_diff < self._scripted_close_z_tol)
            & (~self._scripted_belt_stopped() | deep_enough)
        )
        phase = torch.where(to_close, torch.full_like(phase, self._PHASE_CLOSE), phase)

        in_close = phase == self._PHASE_CLOSE
        self._scripted_close_counter = torch.where(
            in_close,
            self._scripted_close_counter + 1,
            torch.zeros_like(self._scripted_close_counter),
        )
        to_lift = self._scripted_close_done(in_close)
        phase = torch.where(to_lift, torch.full_like(phase, self._PHASE_LIFT), phase)

        to_hold = (phase == self._PHASE_LIFT) & (tcp_pos[:, 2] > deck_z + self._scripted_hold_z_above)
        phase = torch.where(to_hold, torch.full_like(phase, self._PHASE_HOLD), phase)

        self._scripted_phase = phase

        if self._scripted_auto_scale is not None:
            endgame = ((phase == self._PHASE_DESCEND) | (phase == self._PHASE_CLOSE)) & (
                torch.abs(self.belt.velocity) > 1e-6
            )
            self._scripted_auto_scale = torch.where(
                endgame,
                self._scripted_auto_scale.clamp(min=self._scripted_grasp_endgame_scale_floor),
                self._scripted_auto_scale,
            )

        target_world = tcp_pos.clone()

        v_xy = rider.linear_velocity[:, :2]
        lead_xy = self._lookahead_xy(v_xy, xy_dist, rider_pos[:, :2] - tcp_pos[:, :2])
        comove_lead = torch.zeros_like(lead_xy)
        comove_lead[:, 1] = self.belt.velocity * self._scripted_close_lead_time
        if self._scripted_auto_scale is not None:
            sign = self.belt.direction_sign
            rel_xy = tcp_pos[:, :2] - rider_pos[:, :2]
            along = rel_xy[:, 1] * sign
            cross = torch.abs(rel_xy[:, 0])
            near = (along.abs() < self._scripted_auto_bias_radius) & (cross < self._scripted_auto_bias_cross)
            bias = torch.where(
                near,
                (self._scripted_auto_lead_bias - self._scripted_auto_bias_gain * along).clamp(
                    0.0, self._scripted_auto_bias_max
                ),
                self._scripted_auto_lead_bias,
            )
            self._scripted_auto_lead_bias = bias
            speed_ramp = _smoothstep(
                (torch.abs(self.belt.velocity) - self._scripted_auto_bias_v_lo)
                / (self._scripted_auto_bias_v_hi - self._scripted_auto_bias_v_lo)
            )
            comove_lead[:, 1] = comove_lead[:, 1] + sign * bias * speed_ramp
        track_xy = rider_pos[:, :2] + lead_xy + comove_lead

        is_approach = phase == self._PHASE_APPROACH
        target_world[:, 0] = torch.where(is_approach, track_xy[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(is_approach, track_xy[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(
            is_approach,
            torch.full_like(target_world[:, 2], deck_z + self._scripted_approach_z_above),
            target_world[:, 2],
        )

        is_descend = phase == self._PHASE_DESCEND
        target_world[:, 0] = torch.where(is_descend, track_xy[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(is_descend, track_xy[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(is_descend, grasp_z, target_world[:, 2])

        is_close = phase == self._PHASE_CLOSE
        close_xy = rider_pos[:, :2] + comove_lead
        target_world[:, 0] = torch.where(is_close, close_xy[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(is_close, close_xy[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(is_close, grasp_z, target_world[:, 2])

        is_lift_or_hold = (phase == self._PHASE_LIFT) | (phase == self._PHASE_HOLD)
        target_world[:, 2] = torch.where(
            is_lift_or_hold,
            torch.full_like(target_world[:, 2], deck_z + self._scripted_lift_z_above),
            target_world[:, 2],
        )

        pos_input = self._position_input(target_world, tcp_pos)

        pre_grasp = (phase == self._PHASE_APPROACH) | (phase == self._PHASE_DESCEND)
        self._update_motion_dir(v_xy, pre_grasp)
        rot_input = self._rotation_input(tcp_q)

        is_open = (phase == self._PHASE_APPROACH) | (phase == self._PHASE_DESCEND)
        open_val, close_val = self._scripted_gripper_values()
        gripper = torch.where(
            is_open,
            torch.full((self.num_envs,), open_val, device=device, dtype=torch.float32),
            torch.full((self.num_envs,), close_val, device=device, dtype=torch.float32),
        ).unsqueeze(-1)

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)


class _InterceptState(NamedTuple):
    target_world: torch.Tensor
    xy_dist: torch.Tensor
    lower_z: torch.Tensor | float
    tcp_pos: torch.Tensor
    tcp_q: torch.Tensor
    direction: torch.Tensor
    speed: torch.Tensor


class _ScriptedInterceptMixin(ScriptedControlMixin):
    _scripted_hover_z_above: float = 0.20

    _scripted_comove_lead_time: float = 0.08
    _scripted_v_align: float = 0.10
    _scripted_lead_z_weight: float = 0.5
    _scripted_max_lead: float = 0.20

    _scripted_cross_scale: float = 0.06
    _scripted_prox_scale: float = 0.06

    _scripted_auto_park_v_lo: float = 0.15
    _scripted_auto_park_v_hi: float = 0.30

    _scripted_auto_bias_gain: float = 0.3
    _scripted_auto_bias_max: float = 0.10
    _scripted_auto_bias_radius: float = 0.10

    _scripted_auto_dip_slip_lo: float = 0.10
    _scripted_auto_dip_slip_hi: float = 0.30

    @property
    def _scripted_rider(self) -> Actor:
        raise NotImplementedError

    def _scripted_wait_lower_z(
        self, deck_z: float, rider_pos: torch.Tensor
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        raise NotImplementedError

    def _scripted_commit_window(self) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        raise NotImplementedError

    def _intercept_state(self) -> _InterceptState:
        device = self.device
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        hover_z = deck_z + self._scripted_hover_z_above

        rider_pos = self._scripted_rider.pose.p
        wait_z, lower_z = self._scripted_wait_lower_z(deck_z, rider_pos)

        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        tcp_q = agent.tcp.pose.q
        tcp_vel = agent.tcp.linear_velocity
        self._update_auto_speed_scale(tcp_pos, rider_pos)

        d = torch.zeros((self.num_envs, 2), device=device)
        d[:, 1] = self.belt.direction_sign
        self._scripted_motion_dir = d
        speed = torch.abs(self.belt.velocity)

        rel_xy = tcp_pos[:, :2] - rider_pos[:, :2]
        cross = torch.abs(rel_xy[:, 0] * d[:, 1] - rel_xy[:, 1] * d[:, 0])
        xy_dist = torch.linalg.norm(rel_xy, dim=-1)

        v_align = self._auto_scaled(self._scripted_v_align)
        max_lead = self._auto_scaled(self._scripted_max_lead)
        align_gap = cross + self._scripted_lead_z_weight * torch.abs(tcp_pos[:, 2] - wait_z)
        lead_dist = speed * self._scripted_comove_lead_time + speed * align_gap / v_align
        if self._scripted_auto_scale is not None:
            along = rel_xy[:, 0] * d[:, 0] + rel_xy[:, 1] * d[:, 1]
            near = (along.abs() < self._scripted_auto_bias_radius) & (cross < self._scripted_cross_scale)
            bias = torch.where(
                near,
                (self._scripted_auto_lead_bias - self._scripted_auto_bias_gain * along).clamp(
                    0.0, self._scripted_auto_bias_max
                ),
                self._scripted_auto_lead_bias,
            )
            self._scripted_auto_lead_bias = bias
            lead_dist = lead_dist + bias
        lead_dist = lead_dist.clamp(max=max_lead)
        aim_xy = rider_pos[:, :2] + d * lead_dist.unsqueeze(-1)

        r_in, r_out = self._scripted_commit_window()
        a_cross = 1.0 - _smoothstep(cross / self._scripted_cross_scale)
        prox = 1.0 - _smoothstep(torch.linalg.norm(tcp_pos[:, :2] - aim_xy, dim=-1) / self._scripted_prox_scale)
        s_B = 1.0 - _smoothstep((xy_dist - r_in) / (r_out - r_in))
        if self._scripted_auto_scale is not None:
            park = _smoothstep(
                (speed - self._scripted_auto_park_v_lo)
                / (self._scripted_auto_park_v_hi - self._scripted_auto_park_v_lo)
            )
            slip = torch.abs(tcp_vel[:, 0] * d[:, 0] + tcp_vel[:, 1] * d[:, 1] - speed)
            crossing = _smoothstep(
                (slip - self._scripted_auto_dip_slip_lo)
                / (self._scripted_auto_dip_slip_hi - self._scripted_auto_dip_slip_lo)
            )
            s_B = s_B * (1.0 - park * crossing)
            s_A = torch.maximum(torch.maximum(a_cross * prox, s_B), park)
        else:
            s_A = torch.maximum(a_cross * prox, s_B)
        z = hover_z + (wait_z - hover_z) * s_A + (lower_z - wait_z) * s_B

        target_world = torch.stack([aim_xy[:, 0], aim_xy[:, 1], z], dim=-1)
        return _InterceptState(target_world, xy_dist, lower_z, tcp_pos, tcp_q, d, speed)


class ScriptedPlaceMixin(_ScriptedInterceptMixin):
    _PHASE_INTERCEPT = 0
    _PHASE_RELEASE = 1
    _PHASE_WITHDRAW = 2

    _scripted_withdraw_z_above: float = 0.25
    _scripted_lower_z_tol: float = 0.012
    _scripted_lower_xy_tol: float = 0.025
    _scripted_release_steps: int = 10
    _scripted_withdraw_rim_margin: float = 0.03

    _scripted_center_r_in: float = 0.02
    _scripted_center_r_out: float = 0.06

    def _scripted_commit_window(self) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        return self._scripted_center_r_in, self._scripted_center_r_out

    def _scripted_rim_z(self, deck_z: float, rider_pos: torch.Tensor) -> torch.Tensor | float:
        raise NotImplementedError

    def _scripted_release_gripper(self) -> torch.Tensor:
        raise NotImplementedError

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_scripted_phase"):
            self._scripted_phase = torch.full((b,), self._PHASE_INTERCEPT, dtype=torch.long, device=device)
            self._scripted_release_counter = torch.zeros((b,), dtype=torch.long, device=device)
        self._reset_scripted_shared(env_idx)

        if env_idx is None:
            self._scripted_phase[:] = self._PHASE_INTERCEPT
            self._scripted_release_counter[:] = 0
        else:
            self._scripted_phase[env_idx] = self._PHASE_INTERCEPT
            self._scripted_release_counter[env_idx] = 0

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_scripted_phase"):
            self.reset_scripted()

        device = self.device
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height

        rider = self._scripted_rider
        rider_pos = rider.pose.p
        rim_z = self._scripted_rim_z(deck_z, rider_pos)

        state = self._intercept_state()
        phase = self._scripted_phase

        target_world = state.tcp_pos.clone()
        is_intercept = phase == self._PHASE_INTERCEPT
        target_world = torch.where(is_intercept.unsqueeze(-1), state.target_world, target_world)

        to_release = (
            is_intercept
            & (torch.abs(state.tcp_pos[:, 2] - state.lower_z) < self._scripted_lower_z_tol)
            & (state.xy_dist < self._scripted_lower_xy_tol)
        )
        phase = torch.where(to_release, torch.full_like(phase, self._PHASE_RELEASE), phase)

        in_release = phase == self._PHASE_RELEASE
        self._scripted_release_counter = torch.where(
            in_release,
            self._scripted_release_counter + 1,
            torch.zeros_like(self._scripted_release_counter),
        )
        to_withdraw = in_release & (self._scripted_release_counter >= self._scripted_release_steps)
        phase = torch.where(to_withdraw, torch.full_like(phase, self._PHASE_WITHDRAW), phase)
        self._scripted_phase = phase

        is_release = phase == self._PHASE_RELEASE
        release_xy = rider_pos[:, :2] + state.direction * (state.speed * self._scripted_comove_lead_time).unsqueeze(-1)
        target_world[:, 0] = torch.where(is_release, release_xy[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(is_release, release_xy[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(is_release, state.lower_z, target_world[:, 2])

        is_withdraw = phase == self._PHASE_WITHDRAW
        withdraw_tracking = is_withdraw & (state.tcp_pos[:, 2] < rim_z + self._scripted_withdraw_rim_margin)
        target_world[:, 0] = torch.where(withdraw_tracking, rider_pos[:, 0], target_world[:, 0])
        target_world[:, 1] = torch.where(withdraw_tracking, rider_pos[:, 1], target_world[:, 1])
        target_world[:, 2] = torch.where(
            is_withdraw,
            torch.full_like(target_world[:, 2], deck_z + self._scripted_withdraw_z_above),
            target_world[:, 2],
        )

        pos_input = self._position_input(target_world, state.tcp_pos)

        rot_input = self._rotation_input(state.tcp_q)

        is_open_phase = is_release | is_withdraw
        _, close_val = self._scripted_gripper_values()
        gripper = torch.where(
            is_open_phase,
            self._scripted_release_gripper(),
            torch.full((self.num_envs,), close_val, device=device, dtype=torch.float32),
        ).unsqueeze(-1)

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)


class ScriptedInsertMixin(_ScriptedInterceptMixin):
    _scripted_center_margin: float = 0.002
    _scripted_center_min_window: float = 0.004

    @property
    def _scripted_socket_half_width(self) -> float:
        raise NotImplementedError

    @property
    def _scripted_socket_chamfer(self) -> float:
        raise NotImplementedError

    @property
    def _scripted_success_clearance(self) -> float | None:
        return None

    def _scripted_commit_window(self) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        r_in = self._scripted_socket_half_width + self._scripted_center_margin
        r_out = r_in + max(self._scripted_socket_chamfer, self._scripted_center_min_window)
        return r_in, r_out

    def _check_commit_window_contract(self) -> None:
        allowed = self._scripted_success_clearance
        if allowed is None:
            return
        socket = float(self._scripted_socket_half_width)
        allowed = float(allowed)
        if socket > allowed + 1e-9:
            raise ValueError(
                f"{type(self).__name__}: scripted commit window is wider than success allows -- "
                f"_scripted_socket_half_width = {socket:.5f} m exceeds the family's own success "
                f"clearance {allowed:.5f} m. That property is contracted to be the socket's CLEAR "
                f"half-width with the held tool's radius ALREADY subtracted (see "
                f"`_scripted_socket_half_width`); returning the raw socket half-width lets the dip "
                f"reach full commit with the tool outside the socket."
            )

    def _scripted_hold_gripper(self) -> torch.Tensor:
        raise NotImplementedError

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        if not getattr(self, "_scripted_window_checked", False):
            self._check_commit_window_contract()
            self._scripted_window_checked = True
        self._reset_scripted_shared(env_idx)

    def compute_scripted_action(self) -> torch.Tensor:
        self._require_ee_delta_pose()
        if not hasattr(self, "_scripted_motion_dir"):
            self.reset_scripted()

        state = self._intercept_state()

        pos_input = self._position_input(state.target_world, state.tcp_pos)
        rot_input = self._rotation_input(state.tcp_q)
        gripper = self._scripted_hold_gripper()

        return torch.cat([pos_input.to(torch.float32), rot_input.to(torch.float32), gripper], dim=-1)
