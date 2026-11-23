from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_edge_recenter.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_edge_recenter.env import ConveyorEdgeRecenterEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticEdgeRecenter-v0", static=True)
@register_env("StaticEdgeRecenter-v0", max_episode_steps=300)
class StaticEdgeRecenterEnv(StaticConveyorMixin, ConveyorEdgeRecenterEnv):
    """Re-seat an overhanging box on a STATIONARY conveyor belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_spawn_y(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        return self._sample_static_xy(env_idx, b)[:, 1]
