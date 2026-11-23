from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_topple.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_topple.env import ConveyorToppleEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticTopple-v0", static=True)
@register_env("StaticTopple-v0", max_episode_steps=300)
class StaticToppleEnv(StaticConveyorMixin, ConveyorToppleEnv):
    """Topple a standing block resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    _topple_static_standoff: float = 0.045

    def _scripted_park_y(self, oy: float, sign: torch.Tensor, block_pos: torch.Tensor) -> torch.Tensor:
        return block_pos[:, 1] + self._topple_static_standoff * sign

    def _sample_block_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._block_half_z
        return pos
