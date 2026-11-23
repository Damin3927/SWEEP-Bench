from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_crush_limit_pick.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_crush_limit_pick.env import ConveyorCrushLimitPickEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticCrushLimitPick-v0", static=True)
@register_env("StaticCrushLimitPick-v0", max_episode_steps=300)
class StaticCrushLimitPickEnv(StaticConveyorMixin, ConveyorCrushLimitPickEnv):
    """Pick a crushable shell off a stationary belt without squeezing it flat."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_shell_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._core_half_z + 0.001
        return pos
