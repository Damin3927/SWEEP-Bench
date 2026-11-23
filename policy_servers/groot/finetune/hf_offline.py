from __future__ import annotations

import functools
import os

import huggingface_hub
from huggingface_hub.hf_api import ModelInfo

_NOT_MISTRAL = "nvidia/Cosmos-Reason2-2B"


def install() -> None:
    if os.environ.get("HF_HUB_OFFLINE", "").lower() not in ("1", "true", "yes", "on"):
        return
    original = huggingface_hub.model_info

    @functools.wraps(original)
    def model_info(repo_id: str, **kwargs: str | bool | None) -> ModelInfo:
        if repo_id == _NOT_MISTRAL:
            return ModelInfo(id=repo_id, tags=None)
        return original(repo_id, **kwargs)

    setattr(huggingface_hub, "model_info", model_info)  # noqa: B010  # runtime patch of a module attribute
