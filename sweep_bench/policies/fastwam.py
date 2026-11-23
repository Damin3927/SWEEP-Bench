import numpy as np

from sweep_bench.data.vintage import INFO_KEY
from sweep_bench.policies.ee_dom import EeDomWebsocketPolicy, validate_declared_camera_vintage


class FastWAMPolicy(EeDomWebsocketPolicy):
    """Adapter for FastWAM DOM-schema fine-tunes (EE-pose action space)."""

    def __init__(self, host: str, port: int, prompt: str = "", replan_every: int = 8) -> None:
        super().__init__(host=host, port=port, prompt=prompt, replan_every=replan_every)
        if self._obs_offsets[-1] != 0 or any(dt >= 0 for dt in self._obs_offsets[:-1]):
            raise ValueError(
                f"FastWAM server published obs_delta_indices={self._obs_offsets}; expected past offsets "
                "followed by the current frame (0)"
            )
        validate_declared_camera_vintage(self._client.get_server_metadata().get(INFO_KEY), expected=self._SCHEMA.name)

    def _select_frames(self) -> list[dict[str, np.ndarray]]:
        last = len(self._history) - 1
        return [
            {**self._history[max(0, min(last, last + dt))], "is_pad": np.asarray(last + dt < 0)}
            for dt in self._obs_offsets
        ]

    @staticmethod
    def _request_from_frames(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray | str]:
        return {
            **EeDomWebsocketPolicy._request_from_frames(frames),
            "history_is_pad": np.array([bool(f["is_pad"]) for f in frames[:-1]], dtype=bool),
        }
