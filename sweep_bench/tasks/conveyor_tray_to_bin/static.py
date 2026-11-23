from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_tray_to_bin.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_tray_to_bin.env import ConveyorTrayToBinEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticTrayToBin-v0", static=True)
@register_env("StaticTrayToBin-v0", max_episode_steps=300)
class StaticTrayToBinEnv(StaticConveyorMixin, ConveyorTrayToBinEnv):
    """Move the part from the tray to the bin, both resting on a stopped belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_pair_xy(
        self, env_idx: torch.Tensor, b: int, pitch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled = self._sample_static_xy(env_idx, b)
        centre = sampled[:, 1]
        return sampled[:, 0], centre + 0.5 * pitch, centre - 0.5 * pitch
