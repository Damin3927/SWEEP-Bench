from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_peg_pull.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_peg_pull.env import ConveyorPegPullEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticPegPull-v0", static=True)
@register_env("StaticPegPull-v0", max_episode_steps=300)
class StaticPegPullEnv(StaticConveyorMixin, ConveyorPegPullEnv):
    """Pull a pre-seated peg out of a socket block resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_block_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._block_half_height + 0.001
        return pos
