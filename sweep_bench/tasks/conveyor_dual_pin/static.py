from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_dual_pin.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_dual_pin.env import ConveyorDualPinEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticDualPin-v0", static=True)
@register_env("StaticDualPin-v0", max_episode_steps=300)
class StaticDualPinEnv(StaticConveyorMixin, ConveyorDualPinEnv):
    """Register a held bracket onto two pins resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_base_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._base_half_height + 0.001
        return pos
