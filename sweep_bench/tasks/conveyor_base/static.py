from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from sweep_bench.tasks.conveyor_base.config import StaticObjectRandomization

if TYPE_CHECKING:
    from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv

    _StaticBase = ConveyorWorkspaceEnv
else:
    _StaticBase = object


class StaticConveyorMixin(_StaticBase):
    conveyor_speed = 0.0

    def _setup_conveyor_motion(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        return torch.zeros(b, device=self.device, dtype=torch.float32)

    def _sample_spawn_phase(self, env_idx: torch.Tensor, b: int) -> torch.Tensor | None:
        return None

    def _sample_static_xy(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        rcfg = self.randomization
        assert isinstance(rcfg, StaticObjectRandomization), (
            f"static task requires a StaticObjectRandomization (got {type(rcfg).__name__})."
        )
        ox, oy, _oz = self.workspace_offset
        device = self.device
        pos = torch.zeros((b, 3), device=device)
        if rcfg.object_position:
            xs = self._episode_rng.uniform(*rcfg.x_range, size=b).astype(np.float32)
            ys = self._episode_rng.uniform(*rcfg.y_range, size=b).astype(np.float32)
            pos[:, 0] = ox + torch.as_tensor(xs, device=device)
            pos[:, 1] = oy + torch.as_tensor(ys, device=device)
        else:
            pos[:, 0] = ox
            pos[:, 1] = oy
        return pos
