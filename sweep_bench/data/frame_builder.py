from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    import torch

    from sweep_bench.data.schemas import BenchSchema
    from sweep_bench.policies.base import Obs


PANDA_FINGER_MAX = 0.04

WRIST_SENSOR_UID = "wrist_camera"

SENSOR_COMPOSITE_LAYOUT: tuple[tuple[str, ...], ...] = (
    ("side_camera_minus_y", "side_camera_plus_y"),
    ("head_camera",),
)

REQUIRED_SENSOR_UIDS: tuple[str, ...] = tuple(uid for row in SENSOR_COMPOSITE_LAYOUT for uid in row) + (
    WRIST_SENSOR_UID,
)


def build_sensor_composite(panels: dict[str, np.ndarray], h: int, w: int) -> np.ndarray:
    rows = len(SENSOR_COMPOSITE_LAYOUT)
    if h % rows:
        raise ValueError(f"composite height {h} does not divide into {rows} rows")
    ph = h // rows
    bands = []
    for row in SENSOR_COMPOSITE_LAYOUT:
        if w % len(row):
            raise ValueError(f"composite width {w} does not divide into the {len(row)} panels of row {row}")
        pw = w // len(row)
        bands.append(np.concatenate([resize_rgb_to_schema(panels[uid], ph, pw) for uid in row], axis=1))
    return np.concatenate(bands, axis=0)


def build_frame(
    obs: Obs,
    dataset_action: np.ndarray,
    *,
    schema: BenchSchema,
    prompt: str,
    env_idx: int = 0,
) -> dict:
    sensor_data = cast("dict[str, dict[str, torch.Tensor]]", obs["sensor_data"])

    raw = {uid: _raw_rgb(sensor_data[uid], env_idx) for row in SENSOR_COMPOSITE_LAYOUT for uid in row}
    sensors = build_sensor_composite(raw, schema.image_height, schema.image_width)
    wrist = resize_rgb_to_schema(
        _raw_rgb(sensor_data[WRIST_SENSOR_UID], env_idx), schema.image_height, schema.image_width
    )

    agent = cast("dict[str, torch.Tensor]", obs["agent"])
    qpos = agent["qpos"]
    joint_position = qpos[env_idx, : schema.arm_dof].detach().cpu().numpy().astype(np.float32)
    gripper_position = np.array([qpos[env_idx, schema.arm_dof].item() / PANDA_FINGER_MAX], dtype=np.float32)

    return {
        "exterior_image_1_left": sensors,
        "wrist_image_left": wrist,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
        "actions": np.asarray(dataset_action, dtype=np.float32),
        "task": prompt,
    }


def _raw_rgb(cam: dict[str, torch.Tensor], env_idx: int) -> np.ndarray:
    return cam["rgb"][env_idx].detach().cpu().numpy().astype(np.uint8)


def resize_rgb_to_schema(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    if rgb.shape[0] == h and rgb.shape[1] == w:
        return rgb
    from PIL import Image

    img = Image.fromarray(rgb)
    img = img.resize((w, h), resample=Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.uint8)
