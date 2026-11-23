from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_fruit_drop.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_fruit_drop.env import ConveyorFruitDropEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticFruitDrop-v0", static=True)
@register_env("StaticFruitDrop-v0", max_episode_steps=300)
class StaticFruitDropEnv(StaticConveyorMixin, ConveyorFruitDropEnv):
    """Drop a held fruit into a bowl resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_bowl_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._bowl_half_height + 0.001
        return pos
