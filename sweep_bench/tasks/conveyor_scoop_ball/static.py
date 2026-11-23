from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_scoop_ball.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_scoop_ball.env import ConveyorScoopBallEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticScoopBall-v0", static=True)
@register_env("StaticScoopBall-v0", max_episode_steps=300)
class StaticScoopBallEnv(StaticConveyorMixin, ConveyorScoopBallEnv):
    """Sweep a held scoop into a ball resting on a stationary belt and lift it."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_ball_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._scoop_ball_radius[env_idx] + 0.001
        return pos
