from __future__ import annotations

import dataclasses
import functools
import logging
import os

import jax
import jax.numpy as jnp
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.sharding as sharding
import optax
from flax import nnx

ENV_VAR = "BENCH_GRAD_ACCUM"
DISABLE_LICM = "--xla_disable_hlo_passes=while-loop-invariant-code-motion,while-loop-expensive-invariant-code-motion"


def steps_from_env() -> int:
    raw = os.environ.get(ENV_VAR) or "1"
    k = int(raw)
    if k < 1:
        raise ValueError(f"{ENV_VAR} must be >= 1, got {raw!r}")
    return k


def micro_batches(x: jax.Array, k: int) -> jax.Array:
    if x.shape[0] % k:
        raise ValueError(f"batch dimension {x.shape[0]} is not divisible by {ENV_VAR}={k}")
    return jnp.swapaxes(x.reshape(x.shape[0] // k, k, *x.shape[1:]), 0, 1)


def _sharded_like(tree, like):
    mesh = sharding._MeshState.active_mesh  # noqa: SLF001  (upstream's own handle on the mesh set_mesh activates)
    if mesh is None:
        return tree
    return jax.lax.with_sharding_constraint(tree, sharding.fsdp_sharding(like, mesh))


def make_train_step(k: int):
    def train_step(config, rng, state, batch):
        train_rng = jax.random.fold_in(rng, state.step)
        diff_state = nnx.DiffState(0, config.trainable_filter)
        params = state.params.filter(config.trainable_filter)

        def loss_fn(model, rng, observation, actions):
            return jnp.mean(model.compute_loss(rng, observation, actions, train=True))

        def grads_of(rng, observation, actions):
            observation, actions = sharding.activation_sharding_constraint((observation, actions))
            model = nnx.merge(state.model_def, state.params)
            model.train()
            loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, rng, observation, actions)
            return loss.astype(jnp.float32), _sharded_like(jax.tree.map(lambda g: g.astype(jnp.float32), grads), params)

        def accumulate(carry, xs):
            index, (observation, actions) = xs
            rng = jax.lax.cond(index == 0, lambda: train_rng, lambda: jax.random.fold_in(train_rng, index))
            return jax.tree.map(jnp.add, carry, grads_of(rng, observation, actions)), None

        if k == 1:
            loss_sum, grad_sum = grads_of(train_rng, *batch)
        else:
            zeros = _sharded_like(jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32), params), params)
            micro = jax.tree.map(functools.partial(micro_batches, k=k), batch)
            (loss_sum, grad_sum), _ = jax.lax.scan(
                accumulate, (jnp.zeros((), jnp.float32), zeros), (jnp.arange(k), micro)
            )
        loss = loss_sum / k
        grads = jax.tree.map(lambda s, p: (s / k).astype(p.dtype), grad_sum, params)

        model = nnx.merge(state.model_def, state.params)
        model.train()
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        nnx.update(model, new_params)
        new_params = nnx.state(model)

        new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
        if state.ema_decay is not None:
            new_state = dataclasses.replace(
                new_state,
                ema_params=jax.tree.map(
                    lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
                ),
            )

        kernel_params = nnx.state(
            model,
            nnx.All(
                nnx.Param,
                nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
                lambda _, x: x.value.ndim > 1,
            ),
        )
        info = {
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "param_norm": optax.global_norm(kernel_params),
        }
        return new_state, info

    return train_step


def patch_train_step(train_module) -> None:
    k = steps_from_env()
    if k == 1:
        return
    if DISABLE_LICM not in os.environ.get("XLA_FLAGS", "").split():
        raise RuntimeError(f"{ENV_VAR}={k} needs XLA_FLAGS to hold {DISABLE_LICM} (see grad_accum.py)")
    train_module.train_step = make_train_step(k)
    logging.info("%s=%d: each batch runs as %d micro-batches, one optimizer step on their mean gradient", ENV_VAR, k, k)
