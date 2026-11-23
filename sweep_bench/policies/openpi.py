import threading
from collections import deque
from typing import ClassVar, cast

import numpy as np
import torch

from sweep_bench.data.frame_builder import SENSOR_COMPOSITE_LAYOUT, build_sensor_composite, resize_rgb_to_schema
from sweep_bench.data.schemas import get_schema
from sweep_bench.data.vintage import INFO_KEY
from sweep_bench.policies.base import Obs
from sweep_bench.policies.websocket import WebsocketPolicy

type _CameraDict = dict[str, torch.Tensor]
type _SensorData = dict[str, _CameraDict]
type _AgentData = dict[str, torch.Tensor]

EXTERIOR_HISTORY_METADATA_KEY = "exterior_history"
ACTION_SPACE_METADATA_KEY = "action_space"
JOINT_POSITION = "joint_position"


def exterior_history_from_metadata(metadata: dict) -> tuple[int, ...]:
    raw = metadata.get(EXTERIOR_HISTORY_METADATA_KEY)
    if raw is None:
        return ()
    if not isinstance(raw, list | tuple) or not all(isinstance(o, int) and not isinstance(o, bool) for o in raw):
        raise ValueError(f"server declared {EXTERIOR_HISTORY_METADATA_KEY}={raw!r}; expected a list of ints")
    offsets = tuple(raw)
    if any(o >= 0 for o in offsets) or list(offsets) != sorted(set(offsets)):
        raise ValueError(f"server declared {EXTERIOR_HISTORY_METADATA_KEY}={offsets}; expected increasing negatives")
    return offsets


class OpenPiDroidPolicy(WebsocketPolicy):
    """Adapter for joint-position DROID checkpoints (modern openpi recommendation)."""

    _PANDA_FINGER_MAX = 0.04

    _SCHEMA = get_schema("droid_panda_v3")

    _PANDA_ARM_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
    _PANDA_ARM_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float32)

    _HISTORY_SLACK = 32

    _ACTION_SPACES: ClassVar[tuple[str, ...]] = (JOINT_POSITION,)

    def __init__(self, host: str, port: int, prompt: str = "", replan_every: int = 8) -> None:
        super().__init__(host=host, port=port, prompt=prompt, replan_every=replan_every)
        served = self._client.get_server_metadata().get(ACTION_SPACE_METADATA_KEY, JOINT_POSITION)
        if served not in self._ACTION_SPACES:
            raise ValueError(
                f"the policy server serves a {served!r} checkpoint, but {type(self).__name__} speaks "
                f"{self._ACTION_SPACES}; evaluate it with the adapter for {served!r}"
            )
        declared = self._client.get_server_metadata().get(INFO_KEY)
        if declared is not None and declared != self._SCHEMA.name:
            raise ValueError(
                f"the policy server serves a {declared!r} checkpoint, but this adapter renders {self._SCHEMA.name!r} "
                f"frames; restart it without BENCH_CAMERA_SCHEMA and a {self._SCHEMA.name!r} checkpoint"
            )
        self._exterior_history = exterior_history_from_metadata(self._client.get_server_metadata())
        self._episode: deque[np.ndarray] = deque(maxlen=1 - min(self._exterior_history, default=0))
        self._stacks: deque[tuple[Obs, np.ndarray]] = deque(maxlen=self._HISTORY_SLACK)
        self._lock = threading.Lock()

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset(env_idx)
        with self._lock:
            self._episode.clear()

    def observe(self, obs: Obs) -> None:
        if not self._exterior_history:
            return
        composite = self._sensor_composite(cast("_SensorData", obs["sensor_data"]))
        with self._lock:
            self._episode.append(composite)
            last = len(self._episode) - 1
            stack = np.stack([self._episode[max(0, last + o)] for o in (*self._exterior_history, 0)])
            self._stacks.append((obs, stack))

    def predict(self, obs: Obs) -> torch.Tensor:
        self.observe(obs)
        return super().predict(obs)

    def infer_chunk(self, obs: Obs) -> torch.Tensor:
        chunk = self._parse_action_chunk(self._client.infer(self._make_request(obs)))
        return torch.as_tensor(chunk, dtype=torch.float32)

    def _make_request(self, obs: Obs) -> dict[str, np.ndarray | str | float]:
        sensor_data = cast("_SensorData", obs["sensor_data"])
        sensors = self._exterior_stack(obs) if self._exterior_history else self._sensor_composite(sensor_data)
        wrist = self._first_rgb(sensor_data["wrist_camera"])

        agent = cast("_AgentData", obs["agent"])
        qpos = agent["qpos"]
        joint_position = qpos[0, :7].detach().cpu().numpy().astype(np.float32)
        gripper_position = np.array([qpos[0, 7].item() / self._PANDA_FINGER_MAX], dtype=np.float32)

        return {
            "observation/exterior_image_1_left": sensors,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": joint_position,
            "observation/gripper_position": gripper_position,
            "prompt": self.prompt,
            **self._speed_field(),
        }

    def _sensor_composite(self, sensor_data: _SensorData) -> np.ndarray:
        raw = {uid: self._raw_rgb(sensor_data[uid]) for row in SENSOR_COMPOSITE_LAYOUT for uid in row}
        return build_sensor_composite(raw, self._SCHEMA.image_height, self._SCHEMA.image_width)

    def _exterior_stack(self, obs: Obs) -> np.ndarray:
        with self._lock:
            stack = next((stack for seen, stack in reversed(self._stacks) if seen is obs), None)
        if stack is None:
            raise RuntimeError("observe(obs) must run before inference on obs when the server uses exterior history")
        return stack

    @staticmethod
    def _raw_rgb(cam: _CameraDict) -> np.ndarray:
        return cam["rgb"][0].detach().cpu().numpy().astype(np.uint8)

    @classmethod
    def _first_rgb(cls, cam: _CameraDict) -> np.ndarray:
        return resize_rgb_to_schema(cls._raw_rgb(cam), cls._SCHEMA.image_height, cls._SCHEMA.image_width)

    def _parse_action_chunk(self, response: dict[str, np.ndarray]) -> np.ndarray:
        chunk = np.array(response["actions"], dtype=np.float32)
        chunk[..., :7] = np.clip(chunk[..., :7], self._PANDA_ARM_LOWER, self._PANDA_ARM_UPPER)
        chunk[..., 7] = np.clip(2 * chunk[..., 7] - 1, -1.0, 1.0)
        return chunk
