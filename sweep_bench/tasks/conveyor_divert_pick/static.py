from __future__ import annotations

import math

import numpy as np
import torch
from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_divert_pick.config import StaticDivertPickConfig
from sweep_bench.tasks.conveyor_divert_pick.env import ConveyorDivertPickEnv
from sweep_bench.tasks.registry import bench_task


@bench_task("StaticDivertPick-v0", static=True)
@register_env("StaticDivertPick-v0", max_episode_steps=300)
class StaticDivertPickEnv(StaticConveyorMixin, ConveyorDivertPickEnv):
    """Pick a part parked mid-diagonal against a plough on a stationary belt."""

    RandomizationConfig = StaticDivertPickConfig

    def _reset_pick_trackers(self, env_idx: torch.Tensor) -> None:
        super()._reset_pick_trackers(env_idx)
        self._plough_diversion[env_idx] = self.lane_diversion(self.target.pose.p[:, :2])[env_idx]

    def _sample_rider_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        oz = self.workspace_offset[2]
        device = self.device
        sign = self.belt.direction_sign[env_idx]
        rail = self.rail_dir(sign)
        normal = self.rail_normal(sign)

        rcfg = self.randomization
        assert isinstance(rcfg, StaticDivertPickConfig), (
            f"static divert-pick requires a StaticDivertPickConfig (got {type(rcfg).__name__})."
        )
        if rcfg.object_position:
            lo, hi = rcfg.y_range
            draw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
        else:
            draw = torch.zeros(b, device=device)

        standoff = self._kerb_half_thickness + self._rider_half
        anchor = self._static_anchor_rail_offset
        center = self._plough_center_xy(sign)
        xy = center + normal * standoff + rail * (anchor + draw).unsqueeze(-1)
        pos = torch.zeros((b, 3), device=device)
        pos[:, :2] = xy
        pos[:, 2] = oz + self._conveyor_deck_height + self._rider_half + 0.001
        return pos

    @property
    def _static_anchor_rail_offset(self) -> float:
        mid_diversion = 0.5 * (self._plough_commit_offset + 2.0 * self._lane_half)
        along_rail = (mid_diversion - self._lane_half) / math.sin(self._plough_angle)
        return 0.5 * self._kerb_lead_in + along_rail

    def _plough_lane_span(self) -> tuple[float, float]:
        rcfg = self.randomization
        assert isinstance(rcfg, StaticDivertPickConfig)
        lo, hi = rcfg.y_range
        sin_a = math.sin(self._plough_angle)
        mid = 0.5 * (self._plough_commit_offset + 2.0 * self._lane_half)
        return (mid + lo * sin_a, mid + hi * sin_a)
