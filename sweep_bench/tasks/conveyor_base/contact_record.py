from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

CONTACT_FORCE: Final[float] = 0.5


@dataclass(frozen=True)
class ContactRecord:
    contact_step: torch.Tensor
    contact_belt_speed: torch.Tensor
    contact_u: torch.Tensor
    contact_offset: torch.Tensor
    closest_step: torch.Tensor
    closest_belt_speed: torch.Tensor
    closest_u: torch.Tensor
    closest_dist: torch.Tensor
    closest_offset: torch.Tensor

    @classmethod
    def unset(cls, elapsed_steps: torch.Tensor) -> ContactRecord:
        nan = torch.full(elapsed_steps.shape, torch.nan, device=elapsed_steps.device)
        return cls(
            contact_step=torch.full_like(elapsed_steps, -1),
            contact_belt_speed=nan.clone(),
            contact_u=nan.clone(),
            contact_offset=nan.clone(),
            closest_step=torch.full_like(elapsed_steps, -1),
            closest_belt_speed=nan.clone(),
            closest_u=nan.clone(),
            closest_dist=torch.full_like(nan, torch.inf),
            closest_offset=nan.clone(),
        )

    def clear(self, env_idx: torch.Tensor) -> None:
        rows = env_idx.to(self.contact_step.device)
        for steps in (self.contact_step, self.closest_step):
            steps[rows] = -1
        for value in (
            self.contact_belt_speed,
            self.contact_u,
            self.contact_offset,
            self.closest_belt_speed,
            self.closest_u,
            self.closest_offset,
        ):
            value[rows] = torch.nan
        self.closest_dist[rows] = torch.inf

    def update(
        self,
        elapsed_steps: torch.Tensor,
        touching: torch.Tensor,
        belt_speed: torch.Tensor,
        u: torch.Tensor,
        dist: torch.Tensor,
        offset: torch.Tensor,
    ) -> None:
        first = touching & (self.contact_step < 0)
        self.contact_step.copy_(torch.where(first, elapsed_steps, self.contact_step))
        self.contact_belt_speed.copy_(torch.where(first, belt_speed, self.contact_belt_speed))
        self.contact_u.copy_(torch.where(first, u, self.contact_u))
        self.contact_offset.copy_(torch.where(first, offset, self.contact_offset))
        closer = dist < self.closest_dist
        self.closest_step.copy_(torch.where(closer, elapsed_steps, self.closest_step))
        self.closest_belt_speed.copy_(torch.where(closer, belt_speed, self.closest_belt_speed))
        self.closest_u.copy_(torch.where(closer, u, self.closest_u))
        self.closest_dist.copy_(torch.where(closer, dist, self.closest_dist))
        self.closest_offset.copy_(torch.where(closer, offset, self.closest_offset))
