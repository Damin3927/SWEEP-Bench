from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_ring_hang_tilt.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_ring_hang_tilt.env import ConveyorRingHangTiltEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticRingHangTilt-v0", static=True)
@register_env("StaticRingHangTilt-v0", max_episode_steps=300)
class StaticRingHangTiltEnv(StaticConveyorMixin, ConveyorRingHangTiltEnv):
    """Hang a held ring onto the oblique post of a stand on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_stand_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._base_half_height + 0.001
        return pos
