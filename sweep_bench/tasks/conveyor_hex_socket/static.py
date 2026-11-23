from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_hex_socket.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_hex_socket.env import ConveyorHexSocketEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    import torch


@bench_task("StaticHexSocket-v0", static=True)
@register_env("StaticHexSocket-v0", max_episode_steps=300)
class StaticHexSocketEnv(StaticConveyorMixin, ConveyorHexSocketEnv):
    """Clock a held hex bit into the socket of a plate on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_plate_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._plate_half_height + 0.001
        return pos
