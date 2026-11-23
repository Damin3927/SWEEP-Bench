from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_clear_arch.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_clear_arch.env import ConveyorClearArchEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticClearArch-v0", static=True)
@register_env("StaticClearArch-v0", max_episode_steps=300)
class StaticClearArchEnv(StaticConveyorMixin, ConveyorClearArchEnv):
    """Load the bin of an arch pallet resting on a stationary belt, then clear the arch."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_pallet_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._plate_half_thickness + 0.001
        return pos
