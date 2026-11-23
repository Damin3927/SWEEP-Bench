from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_spring_hold.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_spring_hold.env import ConveyorSpringHoldEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticSpringHold-v0", static=True)
@register_env("StaticSpringHold-v0", max_episode_steps=300)
class StaticSpringHoldEnv(StaticConveyorMixin, ConveyorSpringHoldEnv):
    """Hold a sprung tab at the marked extension on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_carrier_xy(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        return self._sample_static_xy(env_idx, b)[:, :2]
