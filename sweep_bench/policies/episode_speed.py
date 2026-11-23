from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import gymnasium as gym

    from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv


def episode_belt_speed(env: gym.Env) -> float:
    belts = cast("ConveyorWorkspaceEnv", env.unwrapped)._velocity_belts
    return abs(float(belts[0].velocity[0].item()))
