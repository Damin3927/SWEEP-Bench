from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
    import types

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import ModalityConfig

EMBODIMENT_TAG_NAME = "NEW_EMBODIMENT"
ACTION_HORIZON = 16
STATE_KEYS = ("joint_position", "gripper_position")
WRIST_COLUMN = "wrist_image_left"
LANGUAGE_KEY = "annotation.human.task_description"


def _load_sibling(name: str) -> types.ModuleType:
    path = pathlib.Path(__file__).resolve().parent / f"{name}.py"
    module_name = f"groot_bench_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to locate {name} at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


history = _load_sibling("history")


def video_keys(exterior_history: tuple[int, ...]) -> list[str]:
    exterior_history = history.validate_offsets(exterior_history)
    return [history.EXTERIOR_COLUMN, WRIST_COLUMN, *(history.video_key(o) for o in exterior_history)]


def embodiment_tag() -> EmbodimentTag:
    from gr00t.data.embodiment_tags import EmbodimentTag

    return EmbodimentTag[EMBODIMENT_TAG_NAME]


def modality_config(exterior_history: tuple[int, ...]) -> dict[str, ModalityConfig]:
    from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=video_keys(exterior_history)),
        "state": ModalityConfig(delta_indices=[0], modality_keys=list(STATE_KEYS)),
        "action": ModalityConfig(
            delta_indices=list(range(ACTION_HORIZON)),
            modality_keys=list(STATE_KEYS),
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_position",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_position",
                ),
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
    }


def register(exterior_history: tuple[int, ...]) -> None:
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    MODALITY_CONFIGS[embodiment_tag().value] = modality_config(exterior_history)


def modality_json() -> dict:
    return {
        "state": {
            "joint_position": {"start": 0, "end": 7, "original_key": "joint_position"},
            "gripper_position": {"start": 0, "end": 1, "original_key": "gripper_position"},
        },
        "action": {
            "joint_position": {"start": 0, "end": 7, "original_key": "actions"},
            "gripper_position": {"start": 7, "end": 8, "original_key": "actions"},
        },
        "video": {
            history.EXTERIOR_COLUMN: {"original_key": history.EXTERIOR_COLUMN},
            WRIST_COLUMN: {"original_key": WRIST_COLUMN},
        },
        "annotation": {LANGUAGE_KEY.removeprefix("annotation."): {"original_key": "task_index"}},
    }


class GrootObservation(TypedDict):
    video: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    language: dict[str, list[list[str]]]


def groot_observation(
    exterior: np.ndarray,
    wrist: np.ndarray,
    joint_position: np.ndarray,
    gripper_position: np.ndarray,
    prompt: str,
    exterior_history: tuple[int, ...],
) -> GrootObservation:
    frames = exterior if exterior.ndim == 4 else exterior[None]
    expected_frames = len(exterior_history) + 1
    if frames.shape[0] != expected_frames:
        raise ValueError(
            f"exterior history {exterior_history} needs {expected_frames} exterior frames, got {frames.shape[0]}"
        )
    video = {history.EXTERIOR_COLUMN: frames[-1], WRIST_COLUMN: wrist}
    for offset, frame in zip(exterior_history, frames[:-1], strict=True):
        video[history.video_key(offset)] = frame
    return {
        "video": {k: np.ascontiguousarray(v)[None, None] for k, v in video.items()},
        "state": {"joint_position": joint_position[None, None], "gripper_position": gripper_position[None, None]},
        "language": {LANGUAGE_KEY: [[prompt]]},
    }
