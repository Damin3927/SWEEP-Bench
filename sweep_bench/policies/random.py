import gymnasium as gym
import numpy as np
import torch

from sweep_bench.policies.base import Obs


class RandomPolicy:
    """Sample uniformly from the env's action space."""

    def __init__(self, action_space: gym.Space) -> None:
        self.action_space = action_space

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        pass

    def predict(self, obs: Obs) -> torch.Tensor:
        sample = self.action_space.sample()
        if isinstance(sample, torch.Tensor):
            return sample
        return torch.as_tensor(np.asarray(sample), dtype=torch.float32)
