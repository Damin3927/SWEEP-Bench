from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_cart_couple.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_cart_couple.env import ConveyorCartCoupleEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor


@bench_task("StaticCartCouple-v0", static=True)
@register_env("StaticCartCouple-v0", max_episode_steps=300)
class StaticCartCoupleEnv(StaticConveyorMixin, ConveyorCartCoupleEnv):
    """Push the lead cart back onto the follower's tongue until it couples."""

    RandomizationConfig = StaticRandomizationConfig

    default_prompt: str = "slide the front cart back onto the following cart until they couple, then let go"

    _belt_moving_threshold: float = 0.0

    _cart_spin_tol: float = 0.30

    def _cart_slip(self, cart: Actor, belt_v: torch.Tensor) -> torch.Tensor:
        rel = cart.linear_velocity.clone()
        rel[:, 1] = rel[:, 1] - belt_v
        return torch.linalg.norm(rel, dim=-1)

    def _cart_settled(self, cart: Actor) -> torch.Tensor:
        return torch.linalg.norm(cart.angular_velocity, dim=-1) <= self._cart_spin_tol

    def _sample_cart_xy(
        self, env_idx: torch.Tensor, b: int, gaps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled = self._sample_static_xy(env_idx, b)
        sign = self.belt.direction_sign[env_idx]
        centre = sampled[:, 1]
        return sampled[:, 0], centre + 0.5 * gaps * sign, centre - 0.5 * gaps * sign

    def reset_scripted(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset_scripted(env_idx)
        if not hasattr(self, "_cc_seated_pos"):
            self._cc_seated_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._cc_seated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        if env_idx is None:
            self._cc_seated_pos[:] = 0.0
            self._cc_seated[:] = False
        else:
            self._cc_seated_pos[env_idx] = 0.0
            self._cc_seated[env_idx] = False

    def _cc_arrest_target(self, tcp_pos: torch.Tensor) -> torch.Tensor:
        lead_pos = self.lead.pose.p
        follower_pos = self.follower.pose.p
        push = self._cc_hold_pos.clone()
        push[:, 0] = tcp_pos[:, 0] + (follower_pos[:, 0] - lead_pos[:, 0])
        mated_y = follower_pos[:, 1] + self._mated_center_distance * self.belt.direction_sign
        push[:, 1] = tcp_pos[:, 1] + (mated_y - lead_pos[:, 1])

        fresh = (self._cc_phase == self._CC_RELEASE) & ~self._cc_seated
        self._cc_seated_pos = torch.where(fresh.unsqueeze(-1), tcp_pos, self._cc_seated_pos)
        self._cc_seated = self._cc_seated | fresh
        return torch.where((self._cc_phase == self._CC_ARREST).unsqueeze(-1), push, self._cc_seated_pos)
