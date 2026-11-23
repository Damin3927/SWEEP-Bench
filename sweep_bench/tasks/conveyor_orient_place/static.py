from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_orient_place.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_orient_place.env import ConveyorOrientPlaceEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticOrientPlace-v0", static=True)
@register_env("StaticOrientPlace-v0", max_episode_steps=300)
class StaticOrientPlaceEnv(StaticConveyorMixin, ConveyorOrientPlaceEnv):
    """Place a held block into a yaw-keyed slot resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_tray_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._tray_origin_above_deck
        return pos
