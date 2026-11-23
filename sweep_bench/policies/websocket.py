from typing import TYPE_CHECKING

import numpy as np
import torch
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from sweep_bench.policies.base import Obs

if TYPE_CHECKING:
    from collections.abc import Callable

BELT_SPEED_KEY = "belt_speed"
SPEED_CONDITIONED_KEY = "speed_conditioned"


class WebsocketPolicy:
    """Base class for policies hosted behind openpi's msgpack websocket protocol."""

    def __init__(self, host: str, port: int, prompt: str = "", replan_every: int = 8) -> None:
        self._client = WebsocketClientPolicy(host=host, port=port)
        conditioned = self._client.get_server_metadata().get(SPEED_CONDITIONED_KEY)
        self.server_speed_conditioned: bool | None = None if conditioned is None else bool(conditioned)
        self.belt_speed: float | None = None
        self.speed_source: Callable[[Obs], float] | None = None
        self.prompt = prompt
        self.replan_every = replan_every
        self._chunk: np.ndarray | None = None
        self._step: int = 0

    def _make_request(self, obs: Obs) -> dict[str, np.ndarray | str | float]:
        raise NotImplementedError

    def _speed_field(self) -> dict[str, float]:
        return {} if self.belt_speed is None else {BELT_SPEED_KEY: float(self.belt_speed)}

    def _parse_action_chunk(self, response: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        self._chunk = None
        self._step = 0

    def replan(self) -> None:
        self._chunk = None

    def predict(self, obs: Obs) -> torch.Tensor:
        if self.speed_source is not None:
            self.belt_speed = self.speed_source(obs)
        chunk = self._chunk
        if chunk is None or self._step >= min(self.replan_every, len(chunk)):
            request = self._make_request(obs)
            response = self._client.infer(request)
            chunk = self._parse_action_chunk(response)
            self._chunk = chunk
            self._step = 0

        action_np = chunk[self._step]
        self._step += 1
        return torch.as_tensor(action_np, dtype=torch.float32).unsqueeze(0)
