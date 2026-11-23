from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_flip_lid.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_flip_lid.env import ConveyorFlipLidEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticFlipLid-v0", static=True)
@register_env("StaticFlipLid-v0", max_episode_steps=300)
class StaticFlipLidEnv(StaticConveyorMixin, ConveyorFlipLidEnv):
    """Flip open the hinged lid of a box resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_box_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height
        return pos
