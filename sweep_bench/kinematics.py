import contextlib
import os

import numpy as np
import pytorch_kinematics as pk
import scipy.spatial.transform
import torch
from mani_skill.agents.robots.panda.panda import Panda
from pytorch_kinematics.transforms import matrix_to_quaternion


class PandaFK:
    def __init__(self) -> None:
        with open(Panda.urdf_path, "rb") as f:
            urdf = f.read()
        with (
            open(os.devnull, "w") as fnull,
            contextlib.redirect_stdout(fnull),
            contextlib.redirect_stderr(fnull),
        ):
            self._chain = pk.build_serial_chain_from_urdf(urdf, end_link_name="panda_hand_tcp")

    def tcp_pose(self, arm_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qpos = torch.from_numpy(np.asarray(arm_qpos, dtype=np.float32)).reshape(1, 7)
        matrix = self._chain.forward_kinematics(qpos).get_matrix()
        pos = matrix[0, :3, 3].numpy().astype(np.float32)
        quat = matrix_to_quaternion(matrix[0, :3, :3]).numpy().astype(np.float32)
        return pos, quat


EULER_RX_WRAPPED = "rx_wrapped"
EULER_XZ_WRAPPED = "xz_wrapped"
EULER_CONVENTIONS = (EULER_RX_WRAPPED, EULER_XZ_WRAPPED)


def euler_from_quat_wxyz(quat: np.ndarray, convention: str = EULER_RX_WRAPPED) -> np.ndarray:
    if convention not in EULER_CONVENTIONS:
        raise ValueError(f"unknown euler convention {convention!r}; expected one of {EULER_CONVENTIONS}")
    euler = (
        scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True)
        .as_euler("xyz", degrees=False)
        .astype(np.float32)
    )
    wrapped = [0] if convention == EULER_RX_WRAPPED else [0, 2]
    euler[wrapped] = np.mod(euler[wrapped], 2 * np.pi)
    return euler
