from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_wipe_board.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_wipe_board.env import ConveyorWipeBoardEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticWipeBoard-v0", static=True)
@register_env("StaticWipeBoard-v0", max_episode_steps=300)
class StaticWipeBoardEnv(StaticConveyorMixin, ConveyorWipeBoardEnv):
    """Wipe the marks off a board resting on a stationary belt, then lift off."""

    RandomizationConfig = StaticRandomizationConfig

    default_prompt: str = "wipe the marks off the board on the conveyor belt"

    def _sample_board_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        pos[:, 2] = self.workspace_offset[2] + self._conveyor_deck_height + self._board_top_above_deck
        return pos
