from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, ClassVar, overload

import torch

from sweep_bench.tasks.conveyor_base.env import (
    _PANDA_UIDS,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv

    _ScriptedBase = ConveyorWorkspaceEnv
else:
    _ScriptedBase = object


def _derive_expert_speed_scale(
    gap: torch.Tensor,
    belt_speed: torch.Tensor,
    runway: torch.Tensor,
    *,
    v_ref: float,
    comove_headroom: float,
    runway_frac: float,
    endgame_reserve: float,
    t_budget_min: float,
    v_ceiling: float,
) -> torch.Tensor:
    t_left = runway / belt_speed.clamp(min=1e-6)
    t_budget = (runway_frac * (t_left - endgame_reserve)).clamp(min=t_budget_min)
    v_needed = comove_headroom * belt_speed + gap / t_budget
    return (v_needed / v_ref).clamp(min=1.0, max=v_ceiling / v_ref)


@overload
def _as_col(cap: float | torch.Tensor) -> float | torch.Tensor: ...
@overload
def _as_col(cap: float | torch.Tensor | None) -> float | torch.Tensor | None: ...
def _as_col(cap: float | torch.Tensor | None) -> float | torch.Tensor | None:
    return cap.unsqueeze(-1) if isinstance(cap, torch.Tensor) else cap


def _gate_on_perception_onset(fn: Callable[[ScriptedControlMixin], torch.Tensor]):
    @functools.wraps(fn)
    def wrapper(self: ScriptedControlMixin) -> torch.Tensor:
        return self._apply_perception_onset_hold(fn(self))

    return wrapper


class ScriptedControlMixin(_ScriptedBase):
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        fn = cls.__dict__.get("compute_scripted_action")
        if callable(fn):
            cls.compute_scripted_action = _gate_on_perception_onset(fn)

    _scripted_v_ee_max = 0.15
    _scripted_pos_step = 0.01
    _scripted_rot_gain = 0.3
    _scripted_accel_step: float | None = None

    _velrand_held_scale: torch.Tensor | None = None

    _ROT_INPUT_SCALE = -10.0

    _scripted_motion_dir_min_speed = 0.02

    _scripted_speed_auto: bool = True
    _scripted_auto_comove_headroom: float = 3.0
    _scripted_auto_runway_frac: float = 0.75
    _scripted_auto_endgame_reserve: float = 1.0
    _scripted_auto_t_budget_min: float = 0.4
    _scripted_auto_v_ceiling: float = 2.0
    _scripted_auto_scale: torch.Tensor | None = None

    _scripted_velrand_mean_latent: float = 0.840
    _scripted_velrand_speed_compensation: bool = False
    _scripted_velrand_lead_compensation: bool = False

    _scripted_velrand_floor_top: float = 0.85
    _scripted_velrand_floor_v_lo: float = 0.20
    _scripted_velrand_floor_v_hi: float = 0.50

    _scripted_lookahead_intercept: bool = False
    _scripted_lookahead_realized_fraction: float = 0.38
    _scripted_lookahead_max_eta: float = 2.0
    _scripted_lookahead_unreachable: torch.Tensor | None = None

    _scripted_reach_guard_radius: ClassVar[dict[str, float]] = {
        "panda": 0.90,
        "panda_wristcam": 0.90,
    }

    _scripted_perception_onset_hold: bool = True
    _scripted_onset_seen: torch.Tensor | None = None

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        raise NotImplementedError

    def compute_scripted_action(self) -> torch.Tensor:
        raise NotImplementedError

    def _reset_scripted_shared(self, env_idx: torch.Tensor | None) -> None:
        b = self.num_envs
        device = self.device
        if not hasattr(self, "_scripted_motion_dir"):
            self._scripted_motion_dir = torch.zeros((b, 2), device=device)
            self._velrand_x = torch.full(
                (b, 3),
                float(self.velocity_randomization.target),
                dtype=torch.float32,
                device=device,
            )
            self._scripted_prev_delta = torch.zeros((b, 3), device=device)
            self._scripted_auto_lead_bias = torch.zeros((b,), device=device)
            self._velrand_gen = torch.Generator(device=device)

        sign = self.belt.direction_sign
        if env_idx is None:
            self._velrand_gen.manual_seed(int(self._episode_seed[0]))
            self._scripted_motion_dir[:, 0] = 0.0
            self._scripted_motion_dir[:, 1] = sign
            self._velrand_x[:] = float(self.velocity_randomization.target)
            self._scripted_prev_delta[:] = 0.0
            self._scripted_auto_lead_bias[:] = 0.0
            self._onset_seen_tensor()[:] = False
        else:
            self._scripted_motion_dir[env_idx, 0] = 0.0
            self._scripted_motion_dir[env_idx, 1] = sign[env_idx]
            self._velrand_x[env_idx] = float(self.velocity_randomization.target)
            self._scripted_prev_delta[env_idx] = 0.0
            self._scripted_auto_lead_bias[env_idx] = 0.0
            self._onset_seen_tensor()[env_idx] = False

    def _onset_seen_tensor(self) -> torch.Tensor:
        seen = self._scripted_onset_seen
        if seen is None:
            seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._scripted_onset_seen = seen
        return seen

    def _apply_perception_onset_hold(self, action: torch.Tensor) -> torch.Tensor:
        if not self._scripted_perception_onset_hold:
            return action
        seen = self._onset_seen_tensor() | self.rider_in_onset_view()
        self._scripted_onset_seen = seen

        held = (~seen).unsqueeze(-1)
        pose_channels = action[:, :-1]
        gated = action.clone()
        gated[:, :-1] = torch.where(held, torch.zeros_like(pose_channels), pose_channels)
        self._scripted_prev_delta = torch.where(
            held, torch.zeros_like(self._scripted_prev_delta), self._scripted_prev_delta
        )
        return gated

    def _update_auto_speed_scale(self, tcp_pos: torch.Tensor, rider_pos: torch.Tensor) -> None:
        if not self._scripted_speed_auto:
            self._scripted_auto_scale = None
            return
        sign = self.belt.direction_sign
        belt_speed = torch.abs(self.belt.velocity)
        oy = self.workspace_offset[1]
        runway = (self._conveyor_half_length - (rider_pos[:, 1] - oy) * sign).clamp(min=0.0)
        gap = torch.linalg.norm(rider_pos - tcp_pos, dim=-1)
        self._scripted_auto_scale = _derive_expert_speed_scale(
            gap,
            belt_speed,
            runway,
            v_ref=self._scripted_pos_step * self.control_freq,
            comove_headroom=self._scripted_auto_comove_headroom,
            runway_frac=self._scripted_auto_runway_frac,
            endgame_reserve=self._scripted_auto_endgame_reserve,
            t_budget_min=self._scripted_auto_t_budget_min,
            v_ceiling=self._scripted_auto_v_ceiling,
        )

    @overload
    def _auto_scaled(self, base: float) -> float | torch.Tensor: ...
    @overload
    def _auto_scaled(self, base: float | None) -> float | torch.Tensor | None: ...
    def _auto_scaled(self, base: float | None) -> float | torch.Tensor | None:
        if base is None or self._scripted_auto_scale is None:
            return base
        return base * self._scripted_auto_scale

    def _update_motion_dir(self, v_xy: torch.Tensor, update_mask: torch.Tensor) -> None:
        v_norm = torch.linalg.norm(v_xy, dim=-1, keepdim=True)
        fast_enough = v_norm > self._scripted_motion_dir_min_speed
        update = fast_enough & update_mask.unsqueeze(-1)
        new_dir = v_xy / v_norm.clamp(min=1e-9)
        self._scripted_motion_dir = torch.where(update, new_dir, self._scripted_motion_dir)

    def _gripper_down_target_q(self) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion

        device = self.device
        b = self.num_envs

        d = self._scripted_motion_dir

        perp = torch.stack([d[:, 1], -d[:, 0]], dim=-1)

        y_body_world = torch.zeros((b, 3), device=device)
        y_body_world[:, :2] = perp
        down = torch.zeros((b, 3), device=device)
        down[:, 2] = -1.0

        z_body_world = down
        x_body_world = torch.cross(y_body_world, z_body_world, dim=-1)

        R = torch.stack([x_body_world, y_body_world, z_body_world], dim=-1)
        return matrix_to_quaternion(R)

    def _lookahead_xy(
        self, v_xy: torch.Tensor, xy_dist: torch.Tensor, rel_xy: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._scripted_lookahead_intercept and rel_xy is not None:
            return self._intercept_lead_xy(v_xy, rel_xy)
        speed_xy = torch.linalg.norm(v_xy, dim=-1, keepdim=True)
        v_ee_max = _as_col(self._auto_scaled(self._scripted_v_ee_max)) * self._velrand_lead_factor()
        catch_up = (v_ee_max - speed_xy).clamp(min=0.05)
        eta = (xy_dist.unsqueeze(-1) / catch_up).clamp(max=2.0)
        return v_xy * eta

    def _intercept_lead_xy(self, v_xy: torch.Tensor, rel_xy: torch.Tensor) -> torch.Tensor:
        s = (
            torch.as_tensor(
                _as_col(self._auto_scaled(self._scripted_v_ee_max)),
                dtype=v_xy.dtype,
                device=v_xy.device,
            )
            * self._scripted_lookahead_realized_fraction
        )
        s = s.expand(v_xy.shape[0], 1) if s.dim() else s.reshape(1, 1).expand(v_xy.shape[0], 1)
        v2 = (v_xy * v_xy).sum(-1, keepdim=True)
        a = v2 - s * s
        b = 2.0 * (rel_xy * v_xy).sum(-1, keepdim=True)
        c = (rel_xy * rel_xy).sum(-1, keepdim=True)
        disc = b * b - 4.0 * a * c
        ok = disc >= 0.0
        sq = torch.sqrt(disc.clamp(min=0.0))
        denom = torch.where(a.abs() < 1e-9, torch.full_like(a, 1e-9), 2.0 * a)
        t = (-b - sq) / denom
        linear = a.abs() < 1e-9
        closing = b < 0.0
        t_lin = torch.where(closing, -c / b.clamp(max=-1e-9), torch.full_like(b, -1.0))
        t = torch.where(linear, t_lin, t)
        reachable = ok & (t >= 0.0)
        t_closest = (-b / (2.0 * v2.clamp(min=1e-9))).clamp(min=0.0)
        self._scripted_lookahead_unreachable = (~reachable).squeeze(-1)
        eta = torch.where(reachable, t, t_closest).clamp(min=0.0, max=self._scripted_lookahead_max_eta)
        return v_xy * eta

    def _advance_velrand_step(self) -> torch.Tensor:
        cfg = self.velocity_randomization
        dt = 1.0 / self.control_freq
        x = self._velrand_x
        drift = -cfg.theta * (x - cfg.target) * dt
        var = torch.clamp(2.0 * cfg.theta * cfg.alpha * x * (1.0 - x), min=0.0)
        sigma = torch.sqrt(var)
        noise = torch.randn(x.shape, generator=self._velrand_gen, dtype=x.dtype, device=x.device) * math.sqrt(dt)
        self._velrand_x = (x + drift + sigma * noise).clamp(0.0, 1.0)
        return self._velrand_scale()

    def _velrand_floor_top_effective(self) -> float:
        return min(self._scripted_velrand_floor_top, self.velocity_randomization.max_scale)

    def _velrand_floor(self) -> float | torch.Tensor:
        cfg = self.velocity_randomization
        top = self._velrand_floor_top_effective()
        if top <= cfg.min_scale or not cfg.enabled:
            return cfg.min_scale
        lo, hi = self._scripted_velrand_floor_v_lo, self._scripted_velrand_floor_v_hi
        speed = torch.abs(self.belt.velocity).reshape(-1, 1)
        ramp = ((speed - lo) / (hi - lo)).clamp(0.0, 1.0)
        return cfg.min_scale + ramp * (top - cfg.min_scale)

    def _velrand_mean_scale(self) -> float | torch.Tensor:
        cfg = self.velocity_randomization
        if not cfg.enabled:
            return 1.0
        floor = self._velrand_floor()
        return floor + self._scripted_velrand_mean_latent * (cfg.max_scale - floor)

    def _velrand_speed_comp(self) -> float | torch.Tensor:
        return 1.0 / self._velrand_mean_scale() if self._scripted_velrand_speed_compensation else 1.0

    def _velrand_lead_factor(self) -> float | torch.Tensor:
        if not self._scripted_velrand_lead_compensation:
            return 1.0
        if self._scripted_velrand_speed_compensation:
            return 1.0
        return self._velrand_mean_scale()

    def _velrand_scale(self) -> torch.Tensor:
        cfg = self.velocity_randomization
        floor = self._velrand_floor()
        return floor + self._velrand_x * (cfg.max_scale - floor)

    def _clamp_pos_delta(self, delta_pos: torch.Tensor) -> torch.Tensor:
        comp = self._velrand_speed_comp()
        accel_step = _as_col(self._auto_scaled(self._scripted_accel_step))
        if accel_step is not None:
            accel_step = accel_step * comp
            prev = self._scripted_prev_delta
            accel = torch.as_tensor(accel_step, dtype=delta_pos.dtype, device=delta_pos.device)
            delta_pos = prev + (delta_pos - prev).clamp(-accel, accel)
        pos_step = (
            torch.as_tensor(
                _as_col(self._auto_scaled(self._scripted_pos_step)),
                dtype=delta_pos.dtype,
                device=delta_pos.device,
            )
            * comp
        )
        xy = delta_pos[:, :2]
        xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-9)
        xy = xy * torch.clamp(pos_step / xy_norm, max=1.0)
        z = delta_pos[:, 2:3].clamp(-pos_step, pos_step)
        clamped = torch.cat([xy, z], dim=-1)
        if self.velocity_randomization.enabled:
            held = self._velrand_held_scale
            scale = self._advance_velrand_step() if held is None else held
            clamped = clamped * scale
        self._scripted_prev_delta = clamped
        return clamped

    def _position_input(self, target_world: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
        delta_pos = self._clamp_pos_delta(target_world - tcp_pos)
        delta_pos = self._apply_reach_guard(delta_pos, tcp_pos)
        return torch.clip(delta_pos / 0.1, -1.0, 1.0)

    def _apply_reach_guard(self, delta_pos: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
        radius = self._scripted_reach_guard_radius.get(self.robot_uids)
        if radius is None:
            raise ValueError(
                f"no scripted reach radius for robot_uids={self.robot_uids!r}; "
                f"known: {sorted(self._scripted_reach_guard_radius)}"
            )
        base = self.agent.robot.pose.p
        offset = tcp_pos + delta_pos - base
        dist = torch.linalg.norm(offset, dim=-1, keepdim=True)
        guarded = base + offset * (radius / dist.clamp(min=1e-9)) - tcp_pos
        return torch.where(dist > radius, guarded, delta_pos)

    def _rotation_input(self, tcp_q: torch.Tensor) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import (
            matrix_to_euler_angles,
            quaternion_invert,
            quaternion_multiply,
            quaternion_to_matrix,
        )

        target_q = self._gripper_down_target_q()
        q_err = quaternion_multiply(target_q, quaternion_invert(tcp_q))
        euler_err = matrix_to_euler_angles(quaternion_to_matrix(q_err), "XYZ")
        rot_input = self._ROT_INPUT_SCALE * self._scripted_rot_gain * euler_err
        rot_norm = torch.linalg.norm(rot_input, dim=-1, keepdim=True).clamp(min=1e-9)
        rot_scale = torch.clamp(1.0 / rot_norm, max=1.0)
        return rot_input * rot_scale

    def _scripted_belt_stopped(self) -> torch.Tensor:
        return torch.abs(self.belt.velocity) <= 1e-6

    def _require_ee_delta_pose(self) -> None:
        if self.control_mode != "pd_ee_delta_pose":
            raise RuntimeError(
                "compute_scripted_action requires control_mode='pd_ee_delta_pose'"
                f" (got {self.control_mode!r}). Pass it via gym.make."
            )

    def _scripted_gripper_values(self) -> tuple[float, float]:
        if self.robot_uids in _PANDA_UIDS:
            return 1.0, -1.0
        raise ValueError(f"compute_scripted_action does not support robot_uids={self.robot_uids!r}")
