from __future__ import annotations

import torch
from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_axle_seat.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_axle_seat.env import ConveyorAxleSeatEnv
from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.registry import bench_task


@bench_task("StaticAxleSeat-v0", static=True)
@register_env("StaticAxleSeat-v0", max_episode_steps=300)
class StaticAxleSeatEnv(StaticConveyorMixin, ConveyorAxleSeatEnv):
    """Seat a held rod into the forks of a carrier resting on a stationary belt."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_carrier_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._plate_thickness / 2.0 + 0.001
        return pos

    def _sample_carrier_quat(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        q = torch.zeros((b, 4), device=self.device)
        q[:, 0] = 1.0
        return q
