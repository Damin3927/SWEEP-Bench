from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_shoe_mat.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_shoe_mat.env import ConveyorShoeMatEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticShoeMat-v0", static=True)
@register_env("StaticShoeMat-v0", max_episode_steps=300)
class StaticShoeMatEnv(StaticConveyorMixin, ConveyorShoeMatEnv):
    """Set the held shoe down sole-down on a mat resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_mat_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        pos[:, 2] = self._mat_ride_z
        return pos
