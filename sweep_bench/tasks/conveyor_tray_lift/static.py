from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_tray_lift.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_tray_lift.env import ConveyorTrayLiftEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticTrayLift-v0", static=True)
@register_env("StaticTrayLift-v0", max_episode_steps=300)
class StaticTrayLiftEnv(StaticConveyorMixin, ConveyorTrayLiftEnv):
    """Lift a loaded tray off a stationary belt without spilling its payload."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_tray_center_xy(self, env_idx: torch.Tensor, b: int, upstream: torch.Tensor) -> torch.Tensor:
        return self._sample_static_xy(env_idx, b)[:, :2]
