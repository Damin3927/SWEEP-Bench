from __future__ import annotations

import os

import einops
import jax.numpy as jnp
from openpi.models import pi0 as _pi0

ENV = "BENCH_BATCHED_SIGLIP"
METADATA_KEY = "bench_batched_siglip"
_UPSTREAM_EMBED_PREFIX = _pi0.Pi0.embed_prefix


def enabled() -> bool:
    value = os.environ.get(ENV, "")
    if value not in ("", "0", "1"):
        raise ValueError(f"{ENV} must be 0, 1 or unset, got {value!r}")
    return value == "1"


def _embed_prefix(self: _pi0.Pi0, obs):
    names = list(obs.images)
    batch = obs.state.shape[0]
    image_tokens, _ = self.PaliGemma.img(jnp.concatenate([obs.images[n] for n in names], axis=0), train=False)
    tokens, input_mask, ar_mask = [], [], []
    for i, name in enumerate(names):
        tokens.append(image_tokens[i * batch : (i + 1) * batch])
        input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
        ar_mask += [False] * image_tokens.shape[1]
    if obs.tokenized_prompt is not None:
        tokens.append(self.PaliGemma.llm(obs.tokenized_prompt, method="embed"))
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * obs.tokenized_prompt.shape[1]
    return jnp.concatenate(tokens, axis=1), jnp.concatenate(input_mask, axis=1), jnp.array(ar_mask)


def install(on: bool) -> None:
    _pi0.Pi0.embed_prefix = _embed_prefix if on else _UPSTREAM_EMBED_PREFIX
