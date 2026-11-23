from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_pour_fill.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_pour_fill.env import ConveyorPourFillEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticPourFill-v0", static=True)
@register_env("StaticPourFill-v0", max_episode_steps=300)
class StaticPourFillEnv(StaticConveyorMixin, ConveyorPourFillEnv):
    """Pour the beads from a held jug into a hopper parked on a stopped belt."""

    RandomizationConfig = StaticRandomizationConfig

    default_prompt: str = "pour the beads from the jug into the hopper on the conveyor belt"

    def _sample_hopper_center_xy(self, env_idx: torch.Tensor, b: int, upstream: torch.Tensor | float) -> torch.Tensor:
        return self._sample_static_xy(env_idx, b)[:, :2]
