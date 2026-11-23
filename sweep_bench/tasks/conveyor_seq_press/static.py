from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_seq_press.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_seq_press.env import ConveyorSeqPressEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticSeqPress-v0", static=True)
@register_env("StaticSeqPress-v0", max_episode_steps=300)
class StaticSeqPressEnv(StaticConveyorMixin, ConveyorSeqPressEnv):
    """Press the panel's two buttons in the commanded order, belt off."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_body_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._body_origin_above_deck
        return pos
