from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_belt_swap.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_belt_swap.env import ConveyorBeltSwapEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.utils.structs.pose import Pose


@bench_task("StaticBeltSwap-v0", static=True)
@register_env("StaticBeltSwap-v0", max_episode_steps=300)
class StaticBeltSwapEnv(StaticConveyorMixin, ConveyorBeltSwapEnv):
    """Move a part across from one STOPPED belt onto the parallel one."""

    RandomizationConfig = StaticRandomizationConfig

    def _sample_belt_b_velocity(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        return torch.zeros(b, device=self.device, dtype=torch.float32)

    _rider_along_tol: float = 0.05

    def _rider_disturbed_now(self, decoy_pose: Pose, deck_z: float) -> torch.Tensor:
        d_pos = decoy_pose.p
        lifted = torch.abs(d_pos[:, 2] - (deck_z + self._distractor_half_z)) > self._rider_z_tol
        moved = torch.abs(d_pos[:, :2] - self._distractor_spawn_xy) > torch.as_tensor(
            [self._rider_x_tol, self._rider_along_tol], device=self.device
        )
        return lifted | self._rider_orientation_disturbed(decoy_pose.q) | moved.any(dim=1)

    _rider_rotation_max_deg: float = 10.0

    def _rider_orientation_disturbed(self, decoy_q: torch.Tensor) -> torch.Tensor:
        dot = (decoy_q * self._distractor_spawn_q).sum(dim=-1).abs().clamp(max=1.0)
        return 2.0 * torch.arccos(dot) > math.radians(self._rider_rotation_max_deg)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        self._static_rider_xy = self._sample_static_xy(env_idx, b)[:, :2]
        super()._initialize_objects(env_idx, b)

    def _sample_rider_xy(
        self, env_idx: torch.Tensor, b: int, *, half_fp: torch.Tensor, downstream: float
    ) -> torch.Tensor:
        xy = self._static_rider_xy.clone()
        xy[:, 1] = xy[:, 1] + downstream
        return xy
