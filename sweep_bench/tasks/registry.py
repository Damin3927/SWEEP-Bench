from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv

_EnvT = TypeVar("_EnvT", bound="ConveyorWorkspaceEnv")

DEFAULT_TASK = "ConveyorAppleGrasp-v0"
DEFAULT_STATIC_TASK = "StaticAppleGrasp-v0"


@dataclass(frozen=True)
class BenchTask:
    gym_id: str
    env_cls: type[ConveyorWorkspaceEnv]
    static: bool = False


_REGISTRY: dict[str, BenchTask] = {}


def bench_task(gym_id: str, *, static: bool = False):
    def decorate(cls: type[_EnvT]) -> type[_EnvT]:
        if gym_id in _REGISTRY:
            raise ValueError(f"bench task {gym_id!r} already registered")
        _REGISTRY[gym_id] = BenchTask(gym_id=gym_id, env_cls=cls, static=static)
        return cls

    return decorate


def all_tasks() -> dict[str, BenchTask]:
    return dict(_REGISTRY)


def get_task(gym_id: str) -> BenchTask:
    if gym_id not in _REGISTRY:
        raise ValueError(f"Unknown task {gym_id!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[gym_id]


def resolve_task_id(task: str | None, static: bool) -> str:
    if task is not None:
        return task
    return DEFAULT_STATIC_TASK if static else DEFAULT_TASK


def build_randomization_config(env_cls: type[ConveyorWorkspaceEnv], randomize: bool):
    cfg_cls = env_cls.RandomizationConfig
    if not randomize:
        return cfg_cls()
    knobs = {f.name: True for f in dataclasses.fields(cfg_cls) if isinstance(f.default, bool)}
    return cfg_cls(**knobs)
