from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_decant.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_decant.env import ConveyorDecantEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticDecant-v0", static=True)
@register_env("StaticDecant-v0", max_episode_steps=300)
class StaticDecantEnv(StaticConveyorMixin, ConveyorDecantEnv):
    """Tip a loaded cup out into the bin, from a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_cup_center_xy(self, env_idx: torch.Tensor, b: int, upstream: torch.Tensor) -> torch.Tensor:
        return self._sample_static_xy(env_idx, b)[:, :2]
