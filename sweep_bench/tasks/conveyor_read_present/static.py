from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_read_present.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_read_present.env import ConveyorReadPresentEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticReadPresent-v0", static=True)
@register_env("StaticReadPresent-v0", max_episode_steps=300)
class StaticReadPresentEnv(StaticConveyorMixin, ConveyorReadPresentEnv):
    """Present a marked part from a stationary belt to the reader, then set it back."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_part_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._part_half[env_idx, 2] + 0.001
        return pos
