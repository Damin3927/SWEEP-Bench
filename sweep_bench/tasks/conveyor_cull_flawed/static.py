from __future__ import annotations

import math

import torch
from mani_skill.utils.registration import register_env

from sweep_bench.tasks.conveyor_base.static import StaticConveyorMixin
from sweep_bench.tasks.conveyor_cull_flawed.config import StaticRandomizationConfig
from sweep_bench.tasks.conveyor_cull_flawed.env import ConveyorCullFlawedEnv
from sweep_bench.tasks.registry import bench_task


@bench_task("StaticCullFlawed-v0", static=True)
@register_env("StaticCullFlawed-v0", max_episode_steps=300)
class StaticCullFlawedEnv(StaticConveyorMixin, ConveyorCullFlawedEnv):
    """Cull the FLAWED part off a stopped belt; leave a SOUND one exactly as found."""

    RandomizationConfig = StaticRandomizationConfig

    _sound_clause: str = "otherwise leave it exactly where it is"
    default_prompt: str = (
        "inspect the part on the conveyor belt: if it carries a defect band, put it in the reject bin; "
        "otherwise leave it exactly where it is"
    )

    @property
    def episode_prompts(self) -> list[str]:
        return [
            f"inspect the {name} on the conveyor belt: if it carries a defect band, "
            f"put it in the reject bin; {self._sound_clause}"
            for name in self._part_display
        ]

    _left_as_found_band: float = 0.005
    _pose_rotation_max_deg: float = 15.0
    _sound_hold_steps: int = 240
    _settled_spin_threshold: float = 0.20

    def _reset_cull_trackers(self, env_idx: torch.Tensor) -> None:
        super()._reset_cull_trackers(env_idx)
        if not hasattr(self, "_left_as_found_counter"):
            b, device = self.num_envs, self.device
            self._left_as_found_counter = torch.zeros(b, dtype=torch.long, device=device)
            self._left_as_found_broken = torch.zeros(b, dtype=torch.bool, device=device)
            self._spawn_y = torch.zeros(b, device=device)
        self._left_as_found_counter[env_idx] = 0
        self._left_as_found_broken[env_idx] = False

    def _sample_part_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        pos = self._sample_static_xy(env_idx, b)
        oz = self.workspace_offset[2]
        pos[:, 2] = oz + self._conveyor_deck_height + self._part_half_z[env_idx] + 0.001
        self._spawn_y[env_idx] = pos[:, 1]
        return pos

    def _sound_branch_delivered(self, part_pos: torch.Tensor) -> torch.Tensor:
        deck_z = self.workspace_offset[2] + self._conveyor_deck_height
        _pos, part_vel = self._part_state()
        on_deck = torch.abs(part_pos[:, 2] - self._part_half_z - deck_z) < self._deck_band
        moved = torch.abs(torch.stack([part_pos[:, 0] - self._spawn_x, part_pos[:, 1] - self._spawn_y], dim=-1))
        in_place = torch.all(moved <= self._left_as_found_band, dim=-1)
        at_rest = (torch.linalg.norm(part_vel, dim=-1) < self._settled_speed_threshold) & (
            torch.linalg.norm(self._part_angular_velocity(), dim=-1) < self._settled_spin_threshold
        )
        dot = (self._primary_rider_pose[:, 3:] * self._part_spawn_q).sum(dim=-1).abs().clamp(max=1.0)
        not_rotated = 2.0 * torch.arccos(dot) <= math.radians(self._pose_rotation_max_deg)
        as_found = on_deck & in_place & at_rest & not_rotated
        self._left_as_found_broken = self._left_as_found_broken | ~as_found
        self._left_as_found_counter = self._advance_sustain("cull_left_as_found", self._left_as_found_counter, as_found)
        self._crossed_clean = self._crossed_clean | (
            (self._left_as_found_counter >= self._sound_hold_steps) & ~self._left_as_found_broken
        )
        return self._crossed_clean
