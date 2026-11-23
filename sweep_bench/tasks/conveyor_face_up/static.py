from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_face_up.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_face_up.env import ConveyorFaceUpEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticFaceUp-v0", static=True)
@register_env("StaticFaceUp-v0", max_episode_steps=300)
class StaticFaceUpEnv(StaticConveyorMixin, ConveyorFaceUpEnv):
    """Pick a die off a stationary belt and set it down showing the commanded face."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_die_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._die_half + 0.001
        return pos
