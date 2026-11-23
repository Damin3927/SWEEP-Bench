from typing import Protocol

import torch

type Obs = dict[str, torch.Tensor | "Obs"]


class Policy(Protocol):
    """Common interface for all policies (in-process or remote-via-websocket)."""

    def reset(self, env_idx: torch.Tensor | None = None) -> None: ...

    def predict(self, obs: Obs) -> torch.Tensor: ...
