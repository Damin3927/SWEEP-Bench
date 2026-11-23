from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_nut_rundown.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_nut_rundown.env import ConveyorNutRunDownEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticNutRunDown-v0", static=True)
@register_env("StaticNutRunDown-v0", max_episode_steps=300)
class StaticNutRunDownEnv(StaticConveyorMixin, ConveyorNutRunDownEnv):
    """Run a wing nut down a threaded post on a fixture resting on a still belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_fixture_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height
        return pos
