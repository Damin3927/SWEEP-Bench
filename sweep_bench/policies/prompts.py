from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym


def episode_prompt(policy_name: str, env: gym.Env) -> str | None:
    prompts = getattr(env.unwrapped, "episode_prompts", None)
    return None if prompts is None else str(prompts[0])
