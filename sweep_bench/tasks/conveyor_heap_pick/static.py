from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_heap_pick.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_heap_pick.env import ConveyorHeapPickEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticHeapPick-v0", static=True)
@register_env("StaticHeapPick-v0", max_episode_steps=300)
class StaticHeapPickEnv(StaticConveyorMixin, ConveyorHeapPickEnv):
    """Pick a part out of a close-packed cluster resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_corral_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        pos[:, 2] = self._corral_rest_z + 0.001
        return pos
