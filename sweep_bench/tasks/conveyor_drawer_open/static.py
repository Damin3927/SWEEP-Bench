from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_drawer_open.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_drawer_open.env import ConveyorDrawerOpenEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticDrawerOpen-v0", static=True)
@register_env("StaticDrawerOpen-v0", max_episode_steps=300)
class StaticDrawerOpenEnv(StaticConveyorMixin, ConveyorDrawerOpenEnv):
    """Slide a captive drawer tray open in a sleeve resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_sleeve_xy(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        return self._sample_static_xy(env_idx, b)[:, :2]
