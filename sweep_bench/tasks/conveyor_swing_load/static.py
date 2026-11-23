from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_swing_load.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_swing_load.env import ConveyorSwingLoadEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticSwingLoad-v0", static=True)
@register_env("StaticSwingLoad-v0", max_episode_steps=300)
class StaticSwingLoadEnv(StaticConveyorMixin, ConveyorSwingLoadEnv):
    """Lift a carry frame off a stationary belt and steady its hanging weight."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_frame_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        pos[:, 2] = self.workspace_offset[2] + self._conveyor_deck_height
        return pos
