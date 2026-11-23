from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_turntable_pick.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_turntable_pick.env import ConveyorTurntablePickEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticTurntablePick-v0", static=True)
@register_env("StaticTurntablePick-v0", max_episode_steps=300)
class StaticTurntablePickEnv(StaticConveyorMixin, ConveyorTurntablePickEnv):
    """Pick a part off a turning table parked on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_plinth_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        pos[:, 2] = self.workspace_offset[2] + self._conveyor_deck_height
        return pos
