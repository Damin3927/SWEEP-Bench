from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_apple.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_apple.env import ConveyorAppleGraspEnv
from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticAppleGrasp-v0", static=True)
@register_env("StaticAppleGrasp-v0", max_episode_steps=300)
class StaticAppleGraspEnv(StaticConveyorMixin, ConveyorAppleGraspEnv):
    """Grasp an apple resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_apple_position(self, env_idx: torch.Tensor, b: int, spawn_z: torch.Tensor) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        pos[:, 2] = self.workspace_offset[2] + spawn_z
        return pos
