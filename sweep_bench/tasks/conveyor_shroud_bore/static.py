from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_shroud_bore.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_shroud_bore.env import ConveyorShroudBoreEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticShroudBore-v0", static=True)
@register_env("StaticShroudBore-v0", max_episode_steps=300)
class StaticShroudBoreEnv(StaticConveyorMixin, ConveyorShroudBoreEnv):
    """Insert a held rod into the shrouded oblique port of a block on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_block_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        pos[:, 2] = self.nominal_origin_z
        return pos
