from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torchvision.transforms.v2.functional as F
from lerobot.configs.types import NormalizationMode
from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.normalize import Normalize, Unnormalize
from policies.dynamicvla.modeling_dynamicvla import (
    DynamicVLAPolicy,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_tensor,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


Signature = tuple[int, tuple[tuple[int, ...], ...]]
KVCache = dict[int, dict[str, torch.Tensor]]


def flow_steps(num_steps: int) -> int:
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32)
    time = torch.tensor(1.0, dtype=torch.float32)
    n = 0
    while time >= -dt / 2:
        time += dt
        n += 1
    return n


def check_stats(module: Normalize | Unnormalize) -> None:
    for key, ft in module.features.items():
        mode = module.norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if mode is NormalizationMode.IDENTITY:
            continue
        buffer = getattr(module, "buffer_" + key.replace(".", "_"))
        names = ("mean", "std") if mode is NormalizationMode.MEAN_STD else ("min", "max")
        for name in names:
            if torch.isinf(buffer[name]).any():
                raise ValueError(f"{key} has no {name} stats (inf), as lerobot's normalizer would assert")


def apply_stats(module: Normalize | Unnormalize, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batch = dict(batch)
    inverse = isinstance(module, Unnormalize)
    for key, ft in module.features.items():
        if key not in batch:
            continue
        mode = module.norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if mode is NormalizationMode.IDENTITY:
            continue
        buffer = getattr(module, "buffer_" + key.replace(".", "_"))
        if mode is NormalizationMode.MEAN_STD:
            mean, std = buffer["mean"], buffer["std"]
            batch[key] = batch[key] * std + mean if inverse else (batch[key] - mean) / (std + 1e-8)
        elif mode is NormalizationMode.MIN_MAX:
            lo, hi = buffer["min"], buffer["max"]
            if inverse:
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (hi - lo) + lo
            else:
                batch[key] = (batch[key] - lo) / (hi - lo + 1e-8)
                batch[key] = batch[key] * 2 - 1
        else:
            raise ValueError(mode)
    return batch


@dataclass
class _Static:
    images: dict[str, torch.Tensor]
    state: torch.Tensor
    tokens: torch.Tensor
    token_mask: torch.Tensor
    noise: torch.Tensor
    pinned: dict[str, torch.Tensor]
    out: torch.Tensor | None = None
    out_pinned: torch.Tensor | None = None
    graph: torch.cuda.CUDAGraph | None = None


class GraphPolicy:
    def __init__(self, policy: DynamicVLAPolicy, device: str, warm_prompts: Sequence[str] = ()) -> None:
        cfg = policy.config
        if cfg.adapt_to_pi_aloha or cfg.enable_streaming:
            raise ValueError("--cuda-graph does not support adapt_to_pi_aloha or enable_streaming checkpoints")
        self.policy = policy
        self.model = policy.model
        self.cfg = cfg
        self.device = torch.device(device)
        policy.eval()
        check_stats(policy.normalize_inputs)
        check_stats(policy.unnormalize_outputs)
        self.n_steps = flow_steps(cfg.num_steps)
        self._consts: dict[tuple, torch.Tensor] = {}
        self._dt = torch.tensor(-1.0 / cfg.num_steps, dtype=torch.float32, device=self.device)
        self._one = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self._time = torch.empty((), dtype=torch.float32, device=self.device)
        self._state_probe = torch.zeros(1, cfg.n_obs_steps, cfg.robot_state_feature.shape[0])
        self._static: dict[Signature, _Static] = {}
        default_images = tuple((cfg.n_obs_steps, *ft.shape[-2:], 3) for ft in cfg.image_features.values())
        for prompt in warm_prompts:
            tokens, _ = self._tokenize(prompt)
            self._prepare((tokens.shape[1], default_images), reason=f"warm-up prompt {prompt!r}")

    def _tokenize(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy.prepare_language({"task": [prompt], OBS_STATE: self._state_probe})

    def _const(self, key: tuple, make: Callable[[], torch.Tensor]) -> torch.Tensor:
        value = self._consts.get(key)
        if value is None:
            if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"constant {key} was first needed inside a CUDA graph capture")
            value = self._consts[key] = make()
        return value

    @torch.no_grad()
    def _prepare(self, sig: Signature, reason: str) -> _Static:
        n_tokens, image_shapes = sig
        cfg = self.cfg
        dev = self.device
        images = {
            k: torch.zeros(shape, dtype=torch.uint8, device=dev)
            for k, shape in zip(cfg.image_features, image_shapes, strict=True)
        }
        s = _Static(
            images=images,
            state=torch.zeros(1, cfg.n_obs_steps, cfg.robot_state_feature.shape[0], dtype=torch.float32, device=dev),
            tokens=torch.zeros(1, n_tokens, dtype=torch.long, device=dev),
            token_mask=torch.ones(1, n_tokens, dtype=torch.bool, device=dev),
            noise=torch.zeros(1, cfg.chunk_size, cfg.max_action_dim, dtype=torch.float32, device=dev),
            pinned={},
        )
        if dev.type == "cuda":
            for name, t in (*images.items(), ("state", s.state), ("tokens", s.tokens), ("token_mask", s.token_mask)):
                s.pinned[name] = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._forward(s)
            torch.cuda.current_stream().wait_stream(side)
            s.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(s.graph):
                s.out = self._forward(s)
            s.out_pinned = torch.empty(s.out.shape, dtype=s.out.dtype, pin_memory=True)
            logging.info("captured a CUDA graph for %d prompt tokens, images %s (%s)", n_tokens, image_shapes, reason)
        self._static[sig] = s
        return s

    @torch.no_grad()
    def infer(self, obs: Mapping[str, object]) -> np.ndarray:
        tokens, token_mask = self._tokenize(str(obs["prompt"]))
        arrays = {k: np.ascontiguousarray(obs[k]) for k in self.cfg.image_features}
        state = np.ascontiguousarray(obs[OBS_STATE], dtype=np.float32)
        for k, a in arrays.items():
            if a.dtype != np.uint8 or a.ndim != 4:
                raise ValueError(f"{k}: expected (T, H, W, 3) uint8, got {a.shape} {a.dtype}")
        if state.shape != (self.cfg.n_obs_steps, self.cfg.robot_state_feature.shape[0]):
            raise ValueError(f"{OBS_STATE}: expected ({self.cfg.n_obs_steps}, state_dim), got {state.shape}")
        sig = (tokens.shape[1], tuple(a.shape for a in arrays.values()))
        s = self._static.get(sig)
        if s is None:
            logging.warning("prompt token count / image shapes %s were not warmed up: capturing now", sig)
            s = self._prepare(sig, reason="first request with this signature")
        noise_shape = (1, self.cfg.chunk_size, self.cfg.max_action_dim)
        if self.device.type != "cuda":
            for k, a in arrays.items():
                s.images[k].copy_(torch.from_numpy(a))
            s.state.copy_(torch.from_numpy(state)[None])
            s.tokens.copy_(tokens)
            s.token_mask.copy_(token_mask)
            s.noise.copy_(self.model._sample_noise(noise_shape, self.device))
            return self._forward(s).numpy().copy()
        for k, a in arrays.items():
            s.pinned[k].numpy()[...] = a
            s.images[k].copy_(s.pinned[k], non_blocking=True)
        s.pinned["state"].numpy()[...] = state[None]
        s.state.copy_(s.pinned["state"], non_blocking=True)
        s.pinned["tokens"].copy_(tokens)
        s.tokens.copy_(s.pinned["tokens"], non_blocking=True)
        s.pinned["token_mask"].copy_(token_mask)
        s.token_mask.copy_(s.pinned["token_mask"], non_blocking=True)
        s.noise.copy_(self.model._sample_noise(noise_shape, self.device))
        assert s.graph is not None and s.out is not None and s.out_pinned is not None
        s.graph.replay()
        s.out_pinned.copy_(s.out, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return s.out_pinned.numpy().copy()

    def _forward(self, s: _Static) -> torch.Tensor:
        cfg = self.cfg
        batch = {OBS_STATE: s.state}
        for key, feat in cfg.image_features.items():
            imgs = s.images[key].permute(0, 3, 1, 2).float() / 255.0
            imgs = F.resize(imgs, list(feat.shape[-2:]))
            batch[key] = imgs.contiguous()[None]
        batch = apply_stats(self.policy.normalize_inputs, batch)
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        actions = self._sample_actions(images, img_masks, s.tokens, s.token_mask, state, s.noise)
        actions = actions[:, :, : cfg.action_feature.shape[0]]
        actions = apply_stats(self.policy.unnormalize_outputs, {ACTION: actions})[ACTION]
        if cfg.use_delta_action:
            action_dim = actions.shape[-1] - 1
            actions[..., :action_dim] += s.state[:, -1:, :action_dim]
        return actions[0].float()

    def _sample_actions(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        bsize = state.shape[0]
        prefix_pad_masks, past_key_values = self._sample_vlm_embedding(
            images, img_masks, lang_tokens, lang_masks, state
        )
        x_t = noise
        time = self._time
        time.copy_(self._one)
        for _ in range(self.n_steps):
            expanded_time = time.expand(bsize)
            v_t = self._denoise_step(prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t += self._dt * v_t
            time += self._dt
        return x_t

    def _sample_vlm_embedding(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, KVCache]:
        model = self.model
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = model._get_position_ids(None, prefix_pad_masks)
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.cfg.use_cache,
            fill_kv_cache=True,
        )
        return prefix_pad_masks, past_key_values

    def _denoise_step(
        self, prefix_pad_masks: torch.Tensor, past_key_values: KVCache, x_t: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        model = self.model
        suffix_embs, suffix_pad_masks, suffix_att_masks = self._embed_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = model._get_position_ids(prefix_offsets, suffix_pad_masks)
        outputs_embeds, _ = model.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.cfg.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.cfg.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return model.action_out_proj(suffix_out)

    def _embed_prefix(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model, cfg = self.model, self.cfg
        embs = []
        pad_masks = []
        att_masks = []
        for img, img_mask in zip(images, img_masks, strict=False):
            bsize = img.size(0)
            if cfg.temporal_fusion == "conv":
                img = model.mults_proj(img)
            elif cfg.temporal_fusion == "flat":
                img = img.view(-1, 3, img.shape[2], img.shape[3])
            img_emb = model.vlm_with_expert.embed_image(img)
            img_emb_dim = img_emb.size(-1)
            if cfg.temporal_fusion == "flat":
                img_emb = img_emb.view(bsize, -1, img_emb_dim)
            scale = self._const(
                ("img_scale", img_emb_dim, img_emb.dtype),
                lambda dim=img_emb_dim, ref=img_emb: torch.tensor(dim**0.5, dtype=ref.dtype, device=ref.device),
            )
            img_emb = img_emb * scale
            num_img_embs = img_emb.size(1)
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * (num_img_embs)
        lang_emb = model.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        if state is not None:
            state_emb = model.state_proj(state)
            state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
            embs.append(state_emb)
            states_seq_len = state_emb.shape[1]
            state_mask = torch.ones(state_emb.shape[0], states_seq_len, dtype=torch.bool, device=state_emb.device)
            pad_masks.append(state_mask)
            att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = self._const(
            ("prefix_att", tuple(att_masks)),
            lambda: torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device),
        )[None, :]
        seq_len = pad_masks.shape[1]
        if seq_len < cfg.prefix_length:
            embs = pad_tensor(embs, cfg.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, cfg.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, cfg.prefix_length, pad_value=0)
        att_masks = att_masks.expand(bsize, -1)
        return embs, pad_masks, att_masks

    def _embed_suffix(
        self, noisy_actions: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model, cfg = self.model, self.cfg
        vlm = model.vlm_with_expert
        action_emb = model.action_in_proj(noisy_actions)
        device = action_emb.device
        dtype = action_emb.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep, vlm.expert_hidden_size, cfg.min_period, cfg.max_period, device=device
        )
        time_emb = time_emb.type(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = model.action_time_mlp_in(action_time_emb)
        action_time_emb = torch.nn.functional.silu(action_time_emb)
        action_time_emb = model.action_time_mlp_out(action_time_emb)
        embs = torch.cat([action_time_emb], dim=1)
        bsize, action_time_dim = action_time_emb.shape[:2]
        pad_masks = torch.cat([torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)], dim=1)
        att_masks = self._const(
            ("suffix_att", cfg.chunk_size, embs.dtype),
            lambda: torch.tensor([1] * cfg.chunk_size, dtype=embs.dtype, device=embs.device),
        )
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks
