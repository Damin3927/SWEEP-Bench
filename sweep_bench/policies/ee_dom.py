import threading
from collections import deque
from typing import cast

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from sweep_bench.data.frame_builder import (
    SENSOR_COMPOSITE_LAYOUT,
    build_sensor_composite,
    resize_rgb_to_schema,
)
from sweep_bench.data.schemas import get_schema
from sweep_bench.kinematics import (
    EULER_CONVENTIONS,
    EULER_XZ_WRAPPED,
    PandaFK,
    euler_from_quat_wxyz,
)
from sweep_bench.policies.base import Obs
from sweep_bench.policies.websocket import WebsocketPolicy

type _CameraDict = dict[str, torch.Tensor]
type _SensorData = dict[str, _CameraDict]
type _AgentData = dict[str, torch.Tensor]


def validate_declared_camera_vintage(declared: object, *, expected: str) -> None:
    if isinstance(declared, str) and declared == expected:
        return
    said = "declared no camera vintage at all" if declared is None else f"declared camera vintage {declared!r}"
    raise ValueError(
        f"the policy server {said}, but this adapter renders {expected!r} frames, so the checkpoint would "
        f"be scored against camera geometry it never saw and its success rate would mean nothing. There is "
        f"no adapt path: {expected!r} is the only registered schema "
        f"(sweep_bench/data/schemas.py). Re-collect the source dataset with it and re-train."
    )


def parse_ee_action_chunk(response: dict[str, np.ndarray]) -> np.ndarray:
    chunk = np.array(response["actions"], dtype=np.float32)
    chunk[:, 3:6] = Rotation.from_euler("xyz", chunk[:, 3:6]).as_euler("XYZ")
    chunk[:, 6] = np.clip(2 * chunk[:, 6] - 1, -1.0, 1.0)
    return chunk


class EeDomWebsocketPolicy(WebsocketPolicy):
    """Adapter for servers speaking the DOM (``_ee_dom``) schema."""

    _SCHEMA = get_schema("droid_panda_v3")

    def __init__(self, host: str, port: int, prompt: str = "", replan_every: int = 8) -> None:
        super().__init__(host=host, port=port, prompt=prompt, replan_every=replan_every)
        metadata = self._client.get_server_metadata()
        self._obs_offsets: list[int] = [int(dt) for dt in metadata.get("obs_delta_indices") or [0]]
        self._history: deque[dict[str, np.ndarray]] = deque(maxlen=1 - min(*self._obs_offsets, 0))
        self._lock = threading.Lock()
        self._fk = PandaFK()
        self._euler_convention = str(metadata.get("euler_convention") or EULER_XZ_WRAPPED)
        if self._euler_convention not in EULER_CONVENTIONS:
            raise ValueError(
                f"server declared euler_convention={self._euler_convention!r}; this bench knows {EULER_CONVENTIONS}"
            )

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        super().reset(env_idx)
        with self._lock:
            self._history.clear()

    def observe(self, obs: Obs) -> None:
        with self._lock:
            self._history.append(self._snapshot(obs))

    def infer_chunk(self, obs: Obs) -> torch.Tensor:
        with self._lock:
            frames = self._select_frames()
        request = {**self._request_from_frames(frames), "prompt": self.prompt, **self._speed_field()}
        chunk = self._parse_action_chunk(self._client.infer(request))
        return torch.as_tensor(chunk, dtype=torch.float32)

    def predict(self, obs: Obs) -> torch.Tensor:
        self.observe(obs)
        return super().predict(obs)

    def _snapshot(self, obs: Obs) -> dict[str, np.ndarray]:
        sensor_data = cast("_SensorData", obs["sensor_data"])
        agent = cast("_AgentData", obs["agent"])
        num_envs = int(agent["qpos"].shape[0])
        if num_envs != 1:
            raise ValueError(
                f"{type(self).__name__} received {num_envs} environments; served policies "
                "are single-env (the server takes one observation window and returns one "
                "action chunk). Re-run with --num-envs 1."
            )
        qpos = agent["qpos"][0, :7].detach().cpu().numpy().astype(np.float32)
        pos, quat = self._fk.tcp_pose(qpos)
        raw = {uid: self._raw_rgb(sensor_data[uid]) for row in SENSOR_COMPOSITE_LAYOUT for uid in row}
        return {
            "opst": build_sensor_composite(raw, self._SCHEMA.image_height, self._SCHEMA.image_width),
            "wrist": self._first_rgb(sensor_data["wrist_camera"]),
            "state": np.concatenate([pos, euler_from_quat_wxyz(quat, self._euler_convention)]).astype(np.float32),
        }

    @staticmethod
    def _raw_rgb(cam: _CameraDict) -> np.ndarray:
        return cam["rgb"][0].detach().cpu().numpy().astype(np.uint8)

    @classmethod
    def _first_rgb(cls, cam: _CameraDict) -> np.ndarray:
        return resize_rgb_to_schema(cls._raw_rgb(cam), cls._SCHEMA.image_height, cls._SCHEMA.image_width)

    def _select_frames(self) -> list[dict[str, np.ndarray]]:
        last = len(self._history) - 1
        return [self._history[max(0, min(last, last + dt))] for dt in self._obs_offsets]

    @staticmethod
    def _request_from_frames(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray | str]:
        return {
            "observation.images.opst_cam": np.stack([f["opst"] for f in frames]),
            "observation.images.wrist_cam": np.stack([f["wrist"] for f in frames]),
            "observation.state": np.stack([f["state"] for f in frames]),
        }

    def _make_request(self, obs: Obs) -> dict[str, np.ndarray | str | float]:
        return {**self._request_from_frames(self._select_frames()), "prompt": self.prompt, **self._speed_field()}

    def _parse_action_chunk(self, response: dict[str, np.ndarray]) -> np.ndarray:
        return parse_ee_action_chunk(response)
