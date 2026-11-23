from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_reach.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_reach.env import ConveyorReachEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticReach-v0", static=True)
@register_env("StaticReach-v0", max_episode_steps=300)
class StaticReachEnv(StaticConveyorMixin, ConveyorReachEnv):
    """Reach a target point above a marker resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_marker_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._marker_origin_above_deck
        return pos
