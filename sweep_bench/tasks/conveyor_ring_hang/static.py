from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_ring_hang.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_ring_hang.env import ConveyorRingHangEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticRingHang-v0", static=True)
@register_env("StaticRingHang-v0", max_episode_steps=300)
class StaticRingHangEnv(StaticConveyorMixin, ConveyorRingHangEnv):
    """Hang a held ring onto the post of a stand resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_stand_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._base_half_height + 0.001
        return pos
