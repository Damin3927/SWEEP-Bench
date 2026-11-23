import gymnasium as gym
import torch

from sweep_bench.policies.base import Obs


class ZeroActionPolicy:
    """Always emit a zero action."""

    def __init__(self, action_space: gym.Space, num_envs: int = 1) -> None:
        shape = action_space.shape
        assert shape is not None
        action_dim = shape[-1]
        self._shape = (num_envs, action_dim)

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        pass

    def predict(self, obs: Obs) -> torch.Tensor:
        return torch.zeros(self._shape, dtype=torch.float32)
