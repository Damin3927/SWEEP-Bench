from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_sheet_edge_pick.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_sheet_edge_pick.env import ConveyorSheetEdgePickEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticSheetEdgePick-v0", static=True)
@register_env("StaticSheetEdgePick-v0", max_episode_steps=300)
class StaticSheetEdgePickEnv(StaticConveyorMixin, ConveyorSheetEdgePickEnv):
    """Pinch a framed sheet off a rack resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_rack_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._rack_origin_z
        return pos
