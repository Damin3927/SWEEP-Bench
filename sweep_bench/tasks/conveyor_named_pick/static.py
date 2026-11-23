from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_named_pick.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_named_pick.env import ConveyorNamedPickEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticNamedPick-v0", static=True)
@register_env("StaticNamedPick-v0", max_episode_steps=300)
class StaticNamedPickEnv(StaticConveyorMixin, ConveyorNamedPickEnv):
    """Pick the named rider out of a pair resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_train_center_xy(self, env_idx: torch.Tensor, b: int, upstream: torch.Tensor) -> torch.Tensor:
        return self._sample_static_xy(env_idx, b)[:, :2]
