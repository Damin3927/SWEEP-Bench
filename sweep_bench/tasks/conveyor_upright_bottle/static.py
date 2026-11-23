from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_upright_bottle.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_upright_bottle.env import ConveyorUprightBottleEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticUprightBottle-v0", static=True)
@register_env("StaticUprightBottle-v0", max_episode_steps=300)
class StaticUprightBottleEnv(StaticConveyorMixin, ConveyorUprightBottleEnv):
    """Stand a fallen bottle upright on a STATIONARY belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_bottle_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._rider_half_z[env_idx] + 0.001
        return pos
