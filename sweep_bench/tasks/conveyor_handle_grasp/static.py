from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_handle_grasp.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_handle_grasp.env import ConveyorHandleGraspEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticHandleGrasp-v0", static=True)
@register_env("StaticHandleGrasp-v0", max_episode_steps=300)
class StaticHandleGraspEnv(StaticConveyorMixin, ConveyorHandleGraspEnv):
    """Lift a handled vessel by its bail off a STATIONARY belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_vessel_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        return self._sample_static_xy(env_idx, b)
