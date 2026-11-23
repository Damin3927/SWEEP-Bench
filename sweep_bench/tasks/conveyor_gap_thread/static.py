from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_gap_thread.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_gap_thread.env import ConveyorGapThreadEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticGapThread-v0", static=True)
@register_env("StaticGapThread-v0", max_episode_steps=300)
class StaticGapThreadEnv(StaticConveyorMixin, ConveyorGapThreadEnv):
    """Thread the held rider into a grooved crate resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_crate_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._base_half_z + 0.001
        return pos
