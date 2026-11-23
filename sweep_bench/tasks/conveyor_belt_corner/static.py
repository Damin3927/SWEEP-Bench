from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_belt_corner.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_belt_corner.env import ConveyorBeltCornerEnv
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.utils.structs.pose import Pose


@bench_task("StaticBeltCorner-v0", static=True)
@register_env("StaticBeltCorner-v0", max_episode_steps=300)
class StaticBeltCornerEnv(StaticConveyorMixin, ConveyorBeltCornerEnv):
    """Transfer a part onto the branch deck of a T-junction of STOPPED belts."""

    RandomizationConfig = StaticRandomizationConfig

    def _setup_branch_motion(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        return torch.zeros(b, device=self.device, dtype=torch.float32)

    _rider_along_tol: float = 0.05

    def _decoy_disturbed_now(self, decoy_pose: Pose, deck_z: float) -> torch.Tensor:
        d_pos = decoy_pose.p
        deck_top = self.workspace_offset[2] + self._conveyor_deck_height + self._distractor_half
        d_lifted = torch.abs(d_pos[:, 2] - deck_top) > self._rider_z_tol
        d_moved = torch.abs(d_pos[:, :2] - self._distractor_spawn_xy) > torch.as_tensor(
            [self._rider_x_tol, self._rider_along_tol], device=self.device
        )
        return d_lifted | self._decoy_orientation_disturbed(decoy_pose.q) | d_moved.any(dim=1)

    _rider_rotation_max_deg: float = 10.0

    def _decoy_orientation_disturbed(self, decoy_q: torch.Tensor) -> torch.Tensor:
        dot = (decoy_q * self._distractor_spawn_q).sum(dim=-1).abs().clamp(max=1.0)
        return 2.0 * torch.arccos(dot) > math.radians(self._rider_rotation_max_deg)

    def _sample_part_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._part_half_z[env_idx] + 0.001
        return pos
