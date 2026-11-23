from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_rel_place.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_rel_place.env import ConveyorRelPlaceEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticRelPlace-v0", static=True)
@register_env("StaticRelPlace-v0", max_episode_steps=300)
class StaticRelPlaceEnv(StaticConveyorMixin, ConveyorRelPlaceEnv):
    """Place the held block in front of / behind the anchor on a STILL tray."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_tray_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._floor_half_height + 0.001
        return pos
