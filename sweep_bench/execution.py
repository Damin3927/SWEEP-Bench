from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    import gymnasium as gym
    import torch

    from sweep_bench.policies import Obs, Policy

    class _SimEnv(Protocol):
        num_envs: int
        control_freq: float

    class ChunkInferencePolicy(Policy, Protocol):
        def observe(self, obs: Obs) -> None: ...

        def infer_chunk(self, obs: Obs) -> torch.Tensor: ...


class ExecutorStats(TypedDict):
    inferences: float
    inference_time_s: float
    hold_steps: float
    steps: float


class Executor(Protocol):
    def reset(self, env_idx: torch.Tensor | None = None) -> None: ...

    def act(self, obs: Obs) -> torch.Tensor: ...

    def stats(self) -> ExecutorStats: ...


class SynchronousExecutor:
    def __init__(self, policy: Policy, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._policy = policy
        self._clock = clock
        self._inferences = 0
        self._inference_time_s = 0.0
        self._steps = 0

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        self._policy.reset(env_idx)

    def act(self, obs: Obs) -> torch.Tensor:
        t0 = self._clock()
        action = self._policy.predict(obs)
        self._inference_time_s += self._clock() - t0
        self._inferences += 1
        self._steps += 1
        return action

    def stats(self) -> ExecutorStats:
        return {
            "inferences": float(self._inferences),
            "inference_time_s": self._inference_time_s,
            "hold_steps": 0.0,
            "steps": float(self._steps),
        }


class _ChunkStream:
    def __init__(self) -> None:
        self._chunk: torch.Tensor | None = None
        self._pos = 0
        self._pending: torch.Tensor | None = None
        self._last: torch.Tensor | None = None

    def offer(self, chunk: torch.Tensor, base_t: int = 0) -> None:
        self._pending = chunk

    def has_pending(self) -> bool:
        return self._pending is not None

    def started(self) -> bool:
        return self._chunk is not None

    def at_boundary(self) -> bool:
        return self._chunk is None or self._pos >= self._chunk.shape[0]

    def step(self, t: int = 0) -> tuple[torch.Tensor | None, bool]:
        if self.at_boundary() and self._pending is not None:
            self._chunk = self._pending
            self._pending = None
            self._pos = 0
        if self._chunk is not None and self._pos < self._chunk.shape[0]:
            self._last = self._chunk[self._pos].unsqueeze(0)
            self._pos += 1
            return self._last, False
        return self._last, self._last is not None

    def clear(self) -> None:
        self._chunk = None
        self._pos = 0
        self._pending = None
        self._last = None


class _LAASStream:
    def __init__(self) -> None:
        self._chunk: torch.Tensor | None = None
        self._base = 0
        self._last: torch.Tensor | None = None

    def offer(self, chunk: torch.Tensor, base_t: int) -> None:
        self._chunk = chunk
        self._base = base_t

    def has_pending(self) -> bool:
        return False

    def started(self) -> bool:
        return self._chunk is not None

    def at_boundary(self) -> bool:
        return True

    def step(self, t: int) -> tuple[torch.Tensor | None, bool]:
        if self._chunk is not None:
            row = t - self._base
            if 0 <= row < self._chunk.shape[0]:
                self._last = self._chunk[row].unsqueeze(0)
                return self._last, False
        return self._last, self._last is not None

    def clear(self) -> None:
        self._chunk = None
        self._base = 0
        self._last = None


class _ThreadedChunkExecutor:
    def __init__(
        self,
        policy: ChunkInferencePolicy,
        *,
        continuous: bool,
        control_dt: float,
        laas: bool = False,
        requires_every_chunk_executed: bool = False,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        start: bool = True,
    ) -> None:
        self._policy = policy
        self._continuous = continuous
        self._gated = not continuous or requires_every_chunk_executed
        self._control_dt = control_dt
        self._clock = clock
        self._sleep = sleep
        self._stream = _LAASStream() if laas else _ChunkStream()
        self._cond = threading.Condition()
        self._obs: Obs | None = None
        self._obs_gen = 0
        self._obs_step = 0
        self._ready: tuple[torch.Tensor, int, int] | None = None
        self._gen = 0
        self._want = not self._gated
        self._busy = False
        self._stop = False
        self._next_at: float | None = None
        self._t = 0
        self._inferences = 0
        self._inference_time_s = 0.0
        self._hold_steps = 0
        self._total_steps = 0
        self._thread = threading.Thread(target=self._worker, name="infer-worker", daemon=True)
        if start:
            self._thread.start()

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    def _worker(self) -> None:
        while True:
            with self._cond:
                while not self._stop and (self._obs is None or not self._want):
                    self._cond.wait()
                if self._stop:
                    return
                obs, gen, base = self._obs, self._obs_gen, self._obs_step
                if self._gated:
                    self._want = False
                self._busy = True
            assert obs is not None
            t0 = self._clock()
            chunk = self._policy.infer_chunk(obs)
            dt = self._clock() - t0
            with self._cond:
                self._ready = (chunk, gen, base)
                self._busy = False
                self._inferences += 1
                self._inference_time_s += dt
                self._cond.notify_all()

    def _take_ready_locked(self) -> None:
        if self._ready is None:
            return
        chunk, gen, base = self._ready
        self._ready = None
        if gen == self._gen:
            self._stream.offer(chunk, base)

    def _idle_locked(self) -> bool:
        return not self._busy and self._ready is None and not self._stream.has_pending()

    def _should_request_locked(self) -> bool:
        if not self._gated:
            return False
        return self._idle_locked() and (self._continuous or self._stream.at_boundary())

    def _pace(self) -> None:
        now = self._clock()
        if self._next_at is not None and self._next_at > now:
            self._sleep(self._next_at - now)
            now = self._clock()
        self._next_at = now + self._control_dt

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        with self._cond:
            self._gen += 1
            self._obs = None
            self._ready = None
            self._stream.clear()
            self._want = not self._gated
            self._next_at = None
            self._t = 0
            self._obs_step = 0
        self._policy.reset(env_idx)

    def act(self, obs: Obs) -> torch.Tensor:
        self._pace()
        self._policy.observe(obs)
        with self._cond:
            self._obs = obs
            self._obs_gen = self._gen
            self._obs_step = self._t
            self._take_ready_locked()
            if self._should_request_locked():
                self._want = True
            self._cond.notify_all()
            while not self._stream.started() and not self._stream.has_pending():
                if self._idle_locked():
                    self._want = True
                self._cond.notify_all()
                self._cond.wait()
                self._take_ready_locked()
        action, held = self._stream.step(self._t)
        self._t += 1
        self._total_steps += 1
        self._hold_steps += held
        assert action is not None
        return action

    def stats(self) -> ExecutorStats:
        with self._cond:
            inferences, time_s = self._inferences, self._inference_time_s
        return {
            "inferences": float(inferences),
            "inference_time_s": time_s,
            "hold_steps": float(self._hold_steps),
            "steps": float(self._total_steps),
        }


class _DeterministicChunkExecutor:
    def __init__(
        self,
        policy: ChunkInferencePolicy,
        *,
        continuous: bool,
        steps: int,
        control_dt: float,
        laas: bool = False,
        requires_every_chunk_executed: bool = False,
    ) -> None:
        self._policy = policy
        self._continuous = continuous
        self._chunk_cadence = requires_every_chunk_executed
        self._latency_steps = steps
        self._control_dt = control_dt
        self._stream = _LAASStream() if laas else _ChunkStream()
        self._t = 0
        self._inflight: torch.Tensor | None = None
        self._inflight_base = 0
        self._land_at = 0
        self._inferences = 0
        self._hold_steps = 0
        self._total_steps = 0

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        self._policy.reset(env_idx)
        self._stream.clear()
        self._t = 0
        self._inflight = None
        self._inflight_base = 0
        self._land_at = 0

    def _infer(self, obs: Obs) -> torch.Tensor:
        self._inferences += 1
        return self._policy.infer_chunk(obs)

    def _land_if_due(self) -> None:
        if self._inflight is not None and self._t >= self._land_at:
            self._stream.offer(self._inflight, self._inflight_base)
            self._inflight = None

    def _dispatch(self, obs: Obs) -> None:
        self._inflight = self._infer(obs)
        self._inflight_base = self._t
        self._land_at = self._t + self._latency_steps
        self._land_if_due()

    def act(self, obs: Obs) -> torch.Tensor:
        self._policy.observe(obs)
        self._land_if_due()

        idle = self._inflight is None and not self._stream.has_pending()
        if not self._stream.started() and idle:
            self._stream.offer(self._infer(obs), self._t)
        elif self._continuous:
            if idle if self._chunk_cadence else self._inflight is None:
                self._dispatch(obs)
        elif self._stream.at_boundary() and idle:
            self._dispatch(obs)

        action, held = self._stream.step(self._t)
        self._t += 1
        self._total_steps += 1
        self._hold_steps += held
        assert action is not None
        return action

    def stats(self) -> ExecutorStats:
        return {
            "inferences": float(self._inferences),
            "inference_time_s": self._inferences * self._latency_steps * self._control_dt,
            "hold_steps": float(self._hold_steps),
            "steps": float(self._total_steps),
        }


_REGISTRY: dict[str, Callable[[Policy, gym.Env, float | None], Executor]] = {}


def executor(
    name: str,
) -> Callable[
    [Callable[[Policy, gym.Env, float | None], Executor]], Callable[[Policy, gym.Env, float | None], Executor]
]:
    def decorate(
        builder: Callable[[Policy, gym.Env, float | None], Executor],
    ) -> Callable[[Policy, gym.Env, float | None], Executor]:
        if name in _REGISTRY:
            raise ValueError(f"executor {name!r} already registered")
        _REGISTRY[name] = builder
        return builder

    return decorate


def all_executors() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def make_executor(name: str, policy: Policy, env: gym.Env, *, inference_latency: float | None = None) -> Executor:
    if name not in _REGISTRY:
        raise ValueError(f"unknown executor {name!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name](policy, env, inference_latency)


@executor("sync")
def _build_sync(policy: Policy, env: gym.Env, inference_latency: float | None) -> Executor:
    return SynchronousExecutor(policy)


@executor("async")
def _build_async(policy: Policy, env: gym.Env, inference_latency: float | None) -> Executor:
    return _build_chunked(policy, env, inference_latency, continuous=False, name="async")


@executor("concurrent")
def _build_concurrent(policy: Policy, env: gym.Env, inference_latency: float | None) -> Executor:
    return _build_chunked(policy, env, inference_latency, continuous=True, name="concurrent")


@executor("concurrent-laas")
def _build_concurrent_laas(policy: Policy, env: gym.Env, inference_latency: float | None) -> Executor:
    return _build_chunked(policy, env, inference_latency, continuous=True, laas=True, name="concurrent-laas")


def _build_chunked(
    policy: Policy, env: gym.Env, inference_latency: float | None, *, continuous: bool, name: str, laas: bool = False
) -> Executor:
    sim = cast("_SimEnv", env.unwrapped)
    if sim.num_envs != 1:
        raise ValueError(f"executor {name!r} runs a single inference stream; pass --num-envs 1 or use --exec-mode sync")
    if not callable(getattr(policy, "infer_chunk", None)):
        raise ValueError(
            f"executor {name!r} needs a policy exposing infer_chunk(obs) -> (K, action_dim); "
            f"{type(policy).__name__} does not — use --policy openpi, or --exec-mode sync"
        )
    if laas and not getattr(policy, "supports_laas", True):
        raise ValueError(
            f"executor {name!r} (latency-aware action streaming) does not support "
            f"{type(policy).__name__}: its world-model feedback assumes whole-chunk execution, which "
            f"LAAS breaks by discarding stale actions. Use --exec-mode concurrent (no LAAS) for it."
        )
    chunk_policy = cast("ChunkInferencePolicy", policy)
    control_dt = 1.0 / sim.control_freq
    whole_chunks = bool(getattr(policy, "requires_every_chunk_executed", False))
    if inference_latency is None:
        return _ThreadedChunkExecutor(
            chunk_policy,
            continuous=continuous,
            control_dt=control_dt,
            laas=laas,
            requires_every_chunk_executed=whole_chunks,
        )
    return _DeterministicChunkExecutor(
        chunk_policy,
        continuous=continuous,
        steps=round(inference_latency / control_dt),
        control_dt=control_dt,
        laas=laas,
        requires_every_chunk_executed=whole_chunks,
    )
