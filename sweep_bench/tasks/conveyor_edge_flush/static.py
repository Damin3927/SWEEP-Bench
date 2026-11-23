from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_edge_flush.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_edge_flush.env import ConveyorEdgeFlushEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticEdgeFlush-v0", static=True)
@register_env("StaticEdgeFlush-v0", max_episode_steps=300)
class StaticEdgeFlushEnv(StaticConveyorMixin, ConveyorEdgeFlushEnv):
    """Square a slab up against a tray's rail on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_tray_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        return self._sample_static_xy(env_idx, b)[:, :2]
