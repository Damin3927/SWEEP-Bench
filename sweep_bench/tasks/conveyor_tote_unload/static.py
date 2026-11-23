from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_tote_unload.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_tote_unload.env import ConveyorToteUnloadEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticToteUnload-v0", static=True)
@register_env("StaticToteUnload-v0", max_episode_steps=300)
class StaticToteUnloadEnv(StaticConveyorMixin, ConveyorToteUnloadEnv):
    """Extract the part from a tote resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_tote_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        pos[:, 2] = self._tote_rest_z + 0.001
        return pos
