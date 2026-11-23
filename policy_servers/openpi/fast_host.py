from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import jax
import numpy as np
from flax import nnx
from openpi.models import model as _model

if TYPE_CHECKING:
    from openpi.policies import policy as _policy

ENV = "BENCH_FAST_HOST"
METADATA_KEY = "bench_fast_host"
_PIXEL_MAX = np.float32(255.0)


def enabled() -> bool:
    value = os.environ.get(ENV, "")
    if value not in ("", "0", "1"):
        raise ValueError(f"{ENV} must be 0, 1 or unset, got {value!r}")
    return value == "1"


class FastHostPolicy:
    def __init__(self, policy: _policy.Policy) -> None:
        if policy._is_pytorch_model:
            raise ValueError(f"{ENV} is for JAX checkpoints; this policy is PyTorch")
        self._policy = policy
        graphdef, self._state = nnx.split(policy._model)
        sample_kwargs = dict(policy._sample_kwargs)

        def run(
            state: nnx.State, rng: jax.Array, inputs: dict, noise: jax.Array | None, pixel_max: jax.Array
        ) -> tuple[jax.Array, dict]:
            model = nnx.merge(graphdef, state)
            rng, sample_rng = jax.random.split(rng)
            batch = jax.tree.map(lambda x: x[np.newaxis, ...], inputs)
            batch["image"] = {
                k: v.astype(np.float32) / pixel_max * 2.0 - 1.0 if v.dtype == np.uint8 else v
                for k, v in batch["image"].items()
            }
            kwargs = sample_kwargs if noise is None else {**sample_kwargs, "noise": noise}
            actions = model.sample_actions(sample_rng, _model.Observation.from_dict(batch), **kwargs)
            return rng, {"state": batch["state"][0], "actions": actions[0]}

        self._run = jax.jit(run)
        self._rng = policy._rng

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        inputs = self._policy._input_transform(jax.tree.map(lambda x: x, obs))
        if noise is not None:
            noise = np.asarray(noise)
            if noise.ndim == 2:
                noise = noise[np.newaxis, ...]
        start_time = time.monotonic()
        self._rng, outputs = self._run(self._state, self._rng, inputs, noise, _PIXEL_MAX)
        model_time = time.monotonic() - start_time
        outputs = self._policy._output_transform(jax.device_get(outputs))
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    @property
    def metadata(self) -> dict[str, object]:
        return self._policy.metadata
