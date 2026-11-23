from typing import ClassVar

import numpy as np

from sweep_bench.kinematics import EULER_CONVENTIONS, PandaFK, euler_from_quat_wxyz
from sweep_bench.policies.base import Obs
from sweep_bench.policies.ee_dom import parse_ee_action_chunk
from sweep_bench.policies.openpi import OpenPiDroidPolicy

EE_ACTION_SPACES = ("ee_pose", "ee_pose_delta")
EULER_METADATA_KEY = "euler_convention"


class OpenPiEePolicy(OpenPiDroidPolicy):
    """Adapter for pi05 checkpoints fine-tuned on the end-effector pose (policy_servers/openpi/finetune/action_space.py)."""

    _ACTION_SPACES: ClassVar[tuple[str, ...]] = EE_ACTION_SPACES

    def __init__(self, host: str, port: int, prompt: str = "", replan_every: int = 8) -> None:
        super().__init__(host=host, port=port, prompt=prompt, replan_every=replan_every)
        convention = self._client.get_server_metadata().get(EULER_METADATA_KEY)
        if convention not in EULER_CONVENTIONS:
            raise ValueError(
                f"the EE server declared {EULER_METADATA_KEY}={convention!r}; expected one of {EULER_CONVENTIONS}"
            )
        self._euler_convention = str(convention)
        self._fk = PandaFK()

    def _make_request(self, obs: Obs) -> dict[str, np.ndarray | str | float]:
        request = super()._make_request(obs)
        pos, quat = self._fk.tcp_pose(np.asarray(request.pop("observation/joint_position"), dtype=np.float32))
        euler = euler_from_quat_wxyz(quat, self._euler_convention)
        return {**request, "observation/ee_pose": np.concatenate([pos, euler]).astype(np.float32)}

    def _parse_action_chunk(self, response: dict[str, np.ndarray]) -> np.ndarray:
        return parse_ee_action_chunk(response)
