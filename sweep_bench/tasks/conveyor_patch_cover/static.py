from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_patch_cover.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_patch_cover.env import ConveyorPatchCoverEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticPatchCover-v0", static=True)
@register_env("StaticPatchCover-v0", max_episode_steps=300)
class StaticPatchCoverEnv(StaticConveyorMixin, ConveyorPatchCoverEnv):
    """Cover the named patch on a STILL board."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_board_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._board_half_z + 0.001
        return pos
