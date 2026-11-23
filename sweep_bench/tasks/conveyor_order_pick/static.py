from __future__ import annotations

import math

import torch
from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_order_pick.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_order_pick.env import ConveyorOrderPickEnv
from sweep_bench.tasks.registry import bench_task


@bench_task("StaticOrderPick-v0", static=True)
@register_env("StaticOrderPick-v0", max_episode_steps=300)
class StaticOrderPickEnv(StaticConveyorMixin, ConveyorOrderPickEnv):
    """Pick the named one of two identical blocks resting on a stopped belt."""

    RandomizationConfig = StaticRandomizationConfig

    PROMPT_TEMPLATES: tuple[str, ...] = (
        "pick up the {ordinal} block from the downstream end of the conveyor belt",
        "grab the {ordinal} block counting back from the conveyor belt's downstream end",
        "take the {ordinal} block off the conveyor belt, counting from its downstream end",
    )

    default_prompt: str = "pick up the first block from the downstream end of the conveyor belt"

    _distractor_along_tol: float = 0.055

    def _distractor_disturbed_now(self, distractor_pos: torch.Tensor, deck_z: float) -> torch.Tensor:
        lifted = distractor_pos[:, 2] > deck_z + self._distractor_rise_threshold
        moved = torch.abs(distractor_pos[:, :2] - self._distractor_spawn_xy) > torch.as_tensor(
            [self._distractor_x_guard, self._distractor_along_tol], device=self.device
        )
        return lifted | self._distractor_orientation_disturbed() | moved.any(dim=1) | self._distractor_moving()

    _distractor_speed_tol: float = 0.02
    _distractor_spin_tol: float = 0.30

    def _distractor_moving(self) -> torch.Tensor:
        return (torch.linalg.norm(self.distractor.linear_velocity, dim=-1) > self._distractor_speed_tol) | (
            torch.linalg.norm(self.distractor.angular_velocity, dim=-1) > self._distractor_spin_tol
        )

    _distractor_rotation_max_deg: float = 15.0

    def _distractor_orientation_disturbed(self) -> torch.Tensor:
        dot = (self.distractor.pose.q * self._distractor_spawn_q).sum(dim=-1).abs().clamp(max=1.0)
        return 2.0 * torch.arccos(dot) > math.radians(self._distractor_rotation_max_deg)

    def _sample_slot_xy(
        self, env_idx: torch.Tensor, b: int, pitch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled = self._sample_static_xy(env_idx, b)
        centre = sampled[:, 1]
        return sampled[:, 0], centre + 0.5 * pitch, centre - 0.5 * pitch
