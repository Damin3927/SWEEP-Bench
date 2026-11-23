from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_pick_place.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_pick_place.env import ConveyorPickPlaceEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticPickPlace-v0", static=True)
@register_env("StaticPickPlace-v0", max_episode_steps=300)
class StaticPickPlaceEnv(StaticConveyorMixin, ConveyorPickPlaceEnv):
    """Pick a cube off a stationary belt and place it into the drop-off bin."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_cube_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._rider_half_z[env_idx] + 0.001
        return pos
