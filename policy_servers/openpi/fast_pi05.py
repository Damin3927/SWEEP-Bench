from __future__ import annotations

import dataclasses
import pathlib
import time
from typing import TYPE_CHECKING

import jax
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F  # noqa: N812
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi_client import base_policy as _base_policy

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_BF16 = torch.bfloat16
_EPS = 1e-6
COMPILE_MODES: dict[str, dict[str, str | int | bool]] = {"default": {}, "max-autotune": {"max_autotune": True}}
_BIG_NEG = -2.3819763e38
_VISION = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
_PROJECTOR = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
_LLM = "paligemma_with_expert.paligemma.model.language_model"
_EXPERT = "paligemma_with_expert.gemma_expert.model"
_TIED_EMBED = "paligemma_with_expert.paligemma.lm_head.weight"
_IMAGE_TOKENS = 256
_IMAGE_SIZE = 224


@dataclasses.dataclass(frozen=True)
class _VisionLayer:
    ln1_w: torch.Tensor
    ln1_b: torch.Tensor
    qkv_w: torch.Tensor
    qkv_b: torch.Tensor
    o_w: torch.Tensor
    o_b: torch.Tensor
    ln2_w: torch.Tensor
    ln2_b: torch.Tensor
    fc1_w: torch.Tensor
    fc1_b: torch.Tensor
    fc2_w: torch.Tensor
    fc2_b: torch.Tensor


@dataclasses.dataclass(frozen=True)
class _GemmaLayer:
    qkv_w: torch.Tensor
    o_w: torch.Tensor
    gate_up_w: torch.Tensor
    down_w: torch.Tensor


@dataclasses.dataclass(frozen=True)
class _StepNorms:
    pre_attn: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    pre_ffw: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    final: tuple[torch.Tensor, torch.Tensor]


def flow_times(num_steps: int) -> list[float]:
    dt = np.float32(-1.0 / num_steps)
    stop = np.float32(-(-1.0 / num_steps) / 2)
    times, t = [], np.float32(1.0)
    while t >= stop:
        times.append(float(t))
        t = np.float32(t + dt)
    return times


def _layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), w, b, _EPS)


def _rms(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + _EPS)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
    b, t, h, d = q.shape
    rows = q.permute(0, 2, 1, 3).reshape(b, h * t, d)
    logits = torch.matmul(rows.float(), k[:, 0].float().transpose(1, 2))
    logits = logits.masked_fill(~key_mask[:, 0], _BIG_NEG)
    probs = torch.softmax(logits, dim=-1).to(_BF16)
    return torch.matmul(probs, v[:, 0]).view(b, h, t, d).transpose(1, 2).reshape(b, t, h * d)


def _vision_layer(x: torch.Tensor, w: _VisionLayer, num_heads: int, query_divisor: torch.Tensor) -> torch.Tensor:
    n, t, c = x.shape
    h = _layer_norm(x, w.ln1_w, w.ln1_b)
    q, k, v = F.linear(h, w.qkv_w, w.qkv_b).view(n, t, 3, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)
    o = F.scaled_dot_product_attention(q / query_divisor, k, v, scale=1.0)
    x = x + F.linear(o.transpose(1, 2).reshape(n, t, c), w.o_w, w.o_b)
    h = F.gelu(F.linear(_layer_norm(x, w.ln2_w, w.ln2_b), w.fc1_w, w.fc1_b), approximate="tanh")
    return x + F.linear(h, w.fc2_w, w.fc2_b)


def _qkv(h: torch.Tensor, w: _GemmaLayer, num_heads: int, head_dim: int, cos, sin):
    b, t, _ = h.shape
    qkv = F.linear(h, w.qkv_w)
    q = qkv[..., : num_heads * head_dim].view(b, t, num_heads, head_dim)
    k = qkv[..., num_heads * head_dim : (num_heads + 1) * head_dim].view(b, t, 1, head_dim)
    v = qkv[..., (num_heads + 1) * head_dim :].view(b, t, 1, head_dim)
    q = _rope(q, cos, sin) * head_dim**-0.5
    return q, _rope(k, cos, sin).transpose(1, 2), v.transpose(1, 2)


def _mlp(h: torch.Tensor, w: _GemmaLayer) -> torch.Tensor:
    gate, up = F.linear(h, w.gate_up_w).chunk(2, dim=-1)
    return F.linear(F.gelu(gate, approximate="tanh") * up, w.down_w)


def _prefix_layer(x, w: _GemmaLayer, pre: torch.Tensor, post: torch.Tensor, cos, sin, k_out, v_out, key_mask, heads):
    q, k, v = _qkv((_rms(x) * pre).to(_BF16), w, heads, k_out.shape[-1], cos, sin)
    k_out.copy_(k)
    v_out.copy_(v)
    x = x + F.linear(_attend(q, k, v, key_mask), w.o_w)
    return x + _mlp((_rms(x) * post).to(_BF16), w)


def _prefix_last_layer_kv(x, w: _GemmaLayer, pre: torch.Tensor, cos, sin, k_out, v_out, heads) -> None:
    _, k, v = _qkv((_rms(x) * pre).to(_BF16), w, heads, k_out.shape[-1], cos, sin)
    k_out.copy_(k)
    v_out.copy_(v)


def _expert_layer(x, w: _GemmaLayer, pre, post, cos, sin, k_cache, v_cache, prefix_len: int, key_mask, heads):
    scale, shift, gate = pre
    q, k, v = _qkv((_rms(x) * scale + shift).to(_BF16), w, heads, k_cache.shape[-1], cos, sin)
    k_cache[:, :, prefix_len:].copy_(k)
    v_cache[:, :, prefix_len:].copy_(v)
    x = x + F.linear(_attend(q, k_cache, v_cache, key_mask), w.o_w) * gate
    scale, shift, gate = post
    return x + _mlp((_rms(x) * scale + shift).to(_BF16), w) * gate


class Pi05Graph:
    def __init__(
        self,
        weights: pathlib.Path | str | dict[str, torch.Tensor],
        *,
        num_images: int,
        max_token_len: int,
        action_horizon: int,
        action_dim: int,
        num_steps: int = 10,
        vision_heads: int = 16,
        device: str = "cuda",
        graph: bool = True,
        compile_mode: str | None = None,
    ) -> None:
        torch.set_float32_matmul_precision("high")
        self.device = torch.device(device)
        self.num_images, self.max_token_len = num_images, max_token_len
        self.action_horizon, self.action_dim = action_horizon, action_dim
        if isinstance(weights, dict):
            sd = {k: v.to(self.device) for k, v in weights.items()}
        else:
            sd = safetensors.torch.load_file(str(weights), device=str(self.device))
        self._load_vision(sd, vision_heads)
        self._load_llm(sd)
        self._load_expert(sd, num_steps)
        del sd
        self.prefix_len = num_images * _IMAGE_TOKENS + max_token_len
        total = self.prefix_len + action_horizon
        layers, d = len(self.llm), self.head_dim
        self.k_cache = torch.zeros(layers, 1, 1, total, d, dtype=_BF16, device=self.device)
        self.v_cache = torch.zeros_like(self.k_cache)
        freq = (2.0 / d) * torch.arange(d // 2, dtype=torch.float32, device=self.device)
        self.timescale = torch.pow(torch.tensor(10_000.0, device=self.device), freq)
        self._vision_layer, self._prefix_layer, self._expert_layer = _vision_layer, _prefix_layer, _expert_layer
        if compile_mode is not None:
            options: dict[str, str | int | bool] = {
                **COMPILE_MODES[compile_mode],
                "emulate_precision_casts": True,
                "triton.cudagraphs": False,
            }

            def compiled(fn):
                return torch.compile(fn, dynamic=False, fullgraph=True, options=options)

            self._vision_layer = compiled(_vision_layer)
            self._prefix_layer = compiled(_prefix_layer)
            self._expert_layer = compiled(_expert_layer)
        self._alloc_io()
        self._graph: torch.cuda.CUDAGraph | None = None
        if graph:
            self._capture()

    def _load_vision(self, sd: dict[str, torch.Tensor], heads: int) -> None:
        emb = f"{_VISION}.embeddings"
        self.patch_w = sd[f"{emb}.patch_embedding.weight"].float()
        self.patch_b = sd[f"{emb}.patch_embedding.bias"].float()
        self.pos_emb = sd[f"{emb}.position_embedding.weight"]
        self.vision: list[_VisionLayer] = []
        i = 0
        while f"{_VISION}.encoder.layers.{i}.layer_norm1.weight" in sd:
            p = f"{_VISION}.encoder.layers.{i}"
            attn = [f"{p}.self_attn.{n}_proj" for n in ("q", "k", "v")]
            self.vision.append(
                _VisionLayer(
                    ln1_w=sd[f"{p}.layer_norm1.weight"],
                    ln1_b=sd[f"{p}.layer_norm1.bias"],
                    qkv_w=torch.cat([sd[f"{a}.weight"] for a in attn]),
                    qkv_b=torch.cat([sd[f"{a}.bias"] for a in attn]),
                    o_w=sd[f"{p}.self_attn.out_proj.weight"],
                    o_b=sd[f"{p}.self_attn.out_proj.bias"],
                    ln2_w=sd[f"{p}.layer_norm2.weight"],
                    ln2_b=sd[f"{p}.layer_norm2.bias"],
                    fc1_w=sd[f"{p}.mlp.fc1.weight"],
                    fc1_b=sd[f"{p}.mlp.fc1.bias"],
                    fc2_w=sd[f"{p}.mlp.fc2.weight"],
                    fc2_b=sd[f"{p}.mlp.fc2.bias"],
                )
            )
            i += 1
        self.post_ln_w = sd[f"{_VISION}.post_layernorm.weight"]
        self.post_ln_b = sd[f"{_VISION}.post_layernorm.bias"]
        self.proj_w = sd[f"{_PROJECTOR}.weight"]
        self.proj_b = sd[f"{_PROJECTOR}.bias"]
        self.vision_heads = heads
        head_dim = self.vision[0].o_w.shape[0] // heads
        self.vision_query_divisor = torch.tensor(float(np.sqrt(head_dim)), dtype=_BF16).to(self.device)

    def _gemma_layers(self, sd: dict[str, torch.Tensor], prefix: str) -> list[_GemmaLayer]:
        layers, i = [], 0
        while f"{prefix}.layers.{i}.self_attn.q_proj.weight" in sd:
            p = f"{prefix}.layers.{i}"
            layers.append(
                _GemmaLayer(
                    qkv_w=torch.cat([sd[f"{p}.self_attn.{n}_proj.weight"] for n in ("q", "k", "v")]),
                    o_w=sd[f"{p}.self_attn.o_proj.weight"],
                    gate_up_w=torch.cat([sd[f"{p}.mlp.gate_proj.weight"], sd[f"{p}.mlp.up_proj.weight"]]),
                    down_w=sd[f"{p}.mlp.down_proj.weight"],
                )
            )
            i += 1
        return layers

    def _load_llm(self, sd: dict[str, torch.Tensor]) -> None:
        embed_key = f"{_LLM}.embed_tokens.weight"
        self.embed = sd[embed_key] if embed_key in sd else sd[_TIED_EMBED]
        width = self.embed.shape[1]
        self.embed_scale = torch.tensor(float(np.sqrt(width)), dtype=_BF16).to(self.device)
        self.llm = self._gemma_layers(sd, _LLM)
        n = len(self.llm)
        self.llm_pre = [1.0 + sd[f"{_LLM}.layers.{i}.input_layernorm.weight"].float() for i in range(n)]
        self.llm_post = [1.0 + sd[f"{_LLM}.layers.{i}.post_attention_layernorm.weight"].float() for i in range(n)]
        self.head_dim = (self.llm[0].qkv_w.shape[0] - self.llm[0].o_w.shape[1]) // 2
        self.heads = self.llm[0].o_w.shape[1] // self.head_dim

    def _load_expert(self, sd: dict[str, torch.Tensor], num_steps: int) -> None:
        self.expert = self._gemma_layers(sd, _EXPERT)
        self.action_in_w = sd["action_in_proj.weight"].float()
        self.action_in_b = sd["action_in_proj.bias"].float()
        self.action_out_w = sd["action_out_proj.weight"]
        self.action_out_b = sd["action_out_proj.bias"]
        self.times = flow_times(num_steps)
        self.dt = torch.tensor(-1.0 / num_steps, dtype=_BF16).to(self.device)
        width = self.action_in_w.shape[0]
        t = torch.tensor(self.times, dtype=torch.float32, device=self.device)
        fraction = torch.linspace(0.0, 1.0, width // 2, dtype=torch.float32, device=self.device)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        angle = t[:, None] * (1.0 / period * 2 * np.pi)
        emb = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        emb = F.silu(F.linear(emb, sd["time_mlp_in.weight"].float(), sd["time_mlp_in.bias"].float()))
        cond = F.silu(F.linear(emb, sd["time_mlp_out.weight"].float(), sd["time_mlp_out.bias"].float())).to(_BF16)

        def modulation(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mod = F.linear(cond, sd[f"{name}.dense.weight"], sd[f"{name}.dense.bias"])
            scale, shift, gate = mod[:, None, None, :].chunk(3, dim=-1)
            return 1.0 + scale.float(), shift.float(), gate

        def over_layers(norm: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mods = [modulation(f"{_EXPERT}.layers.{i}.{norm}") for i in range(len(self.expert))]
            scale, shift, gate = (torch.stack([m[j] for m in mods], dim=1) for j in range(3))
            return scale, shift, gate

        final = modulation(f"{_EXPERT}.norm")
        self.step_norms = _StepNorms(
            pre_attn=over_layers("input_layernorm"),
            pre_ffw=over_layers("post_attention_layernorm"),
            final=(final[0], final[1]),
        )

    def _alloc_io(self) -> None:
        n, length = self.num_images, self.max_token_len
        self.noise_bytes = self.action_horizon * self.action_dim * 4
        self.tokens_at = self.noise_bytes
        self.masks_at = self.tokens_at + length * 4
        self.images_at = -(-(self.masks_at + length + n) // 64) * 64
        size = self.images_at + n * _IMAGE_SIZE * _IMAGE_SIZE * 3
        pin = self.device.type == "cuda"
        self.host_in = torch.zeros(size, dtype=torch.uint8, pin_memory=pin)
        self.host_out = torch.empty(self.action_horizon, self.action_dim, dtype=torch.float32, pin_memory=pin)
        self.dev_in = torch.zeros(size, dtype=torch.uint8, device=self.device)
        self.dev_out = torch.empty(1, self.action_horizon, self.action_dim, dtype=torch.float32, device=self.device)
        self.np_in = self.host_in.numpy()
        self.np_out = self.host_out.numpy()
        self.np_in[self.masks_at] = 1
        self.np_in[self.masks_at + length : self.masks_at + length + n] = 1

    def _forward(self) -> None:
        noise, images, image_mask, tokens, token_mask = self.inputs()
        prefix, prefix_mask = self.embed_prefix(images, image_mask, tokens, token_mask)
        self.prefill(prefix, prefix_mask)
        self.dev_out.copy_(self.denoise(noise, prefix_mask))

    def inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n, length = self.num_images, self.max_token_len
        noise = self.dev_in[: self.noise_bytes].view(torch.float32).view(1, self.action_horizon, self.action_dim)
        tokens = self.dev_in[self.tokens_at : self.masks_at].view(torch.int32).view(1, length)
        token_mask = self.dev_in[self.masks_at : self.masks_at + length].view(1, length) != 0
        image_mask = self.dev_in[self.masks_at + length : self.masks_at + length + n] != 0
        images = self.dev_in[self.images_at :].view(n, _IMAGE_SIZE, _IMAGE_SIZE, 3)
        return noise, images, image_mask, tokens, token_mask

    def embed_prefix(
        self, images: torch.Tensor, image_mask: torch.Tensor, tokens: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = images.shape[0]
        pixels = images.permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0
        x = F.conv2d(pixels, self.patch_w, self.patch_b, stride=self.patch_w.shape[-1])
        x = (x.flatten(2).transpose(1, 2) + self.pos_emb).to(_BF16)
        for w in self.vision:
            x = self._vision_layer(x, w, self.vision_heads, self.vision_query_divisor)
        x = F.linear(_layer_norm(x, self.post_ln_w, self.post_ln_b), self.proj_w, self.proj_b)
        text = F.embedding(tokens, self.embed) * self.embed_scale
        prefix = torch.cat([x.reshape(1, n * _IMAGE_TOKENS, -1), text], dim=1)
        return prefix, torch.cat([image_mask.repeat_interleave(_IMAGE_TOKENS)[None], token_mask], dim=1)

    def prefill(self, prefix: torch.Tensor, prefix_mask: torch.Tensor) -> None:
        cos, sin = self._rope_tables(torch.cumsum(prefix_mask, dim=1) - 1)
        key_mask = prefix_mask[:, None, None, :]
        p, last = self.prefix_len, len(self.llm) - 1
        for i, w in enumerate(self.llm[:last]):
            prefix = self._prefix_layer(
                prefix, w, self.llm_pre[i], self.llm_post[i], cos, sin,
                self.k_cache[i, :, :, :p], self.v_cache[i, :, :, :p], key_mask, self.heads,
            )  # fmt: skip
        _prefix_last_layer_kv(
            prefix, self.llm[last], self.llm_pre[last], cos, sin,
            self.k_cache[last, :, :, :p], self.v_cache[last, :, :, :p], self.heads,
        )  # fmt: skip

    def denoise(self, noise: torch.Tensor, prefix_mask: torch.Tensor) -> torch.Tensor:
        horizon = self.action_horizon
        suffix_positions = prefix_mask.sum(dim=1, keepdim=True) + torch.arange(horizon, device=self.device)[None]
        cos, sin = self._rope_tables(suffix_positions)
        suffix_mask = torch.ones(1, 1, 1, horizon, dtype=torch.bool, device=self.device)
        full_mask = torch.cat([prefix_mask[:, None, None, :], suffix_mask], dim=-1)
        norms = self.step_norms
        x_t = noise
        for step in range(len(self.times)):
            h = F.linear(x_t, self.action_in_w, self.action_in_b).to(_BF16)
            for i, w in enumerate(self.expert):
                h = self._expert_layer(
                    h, w, tuple(t[step, i] for t in norms.pre_attn), tuple(t[step, i] for t in norms.pre_ffw),
                    cos, sin, self.k_cache[i], self.v_cache[i], self.prefix_len, full_mask, self.heads,
                )  # fmt: skip
            h = (_rms(h) * norms.final[0][step] + norms.final[1][step]).to(_BF16)
            v_t = F.linear(h, self.action_out_w, self.action_out_b)
            x_t = x_t + v_t * self.dt
        return x_t

    def _rope_tables(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        radians = positions.float()[..., None, None] / self.timescale
        return torch.cos(radians), torch.sin(radians)

    def _capture(self) -> None:
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.dev_in.copy_(self.host_in, non_blocking=True)
                self._forward()
                self.host_out.copy_(self.dev_out[0], non_blocking=True)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self.dev_in.copy_(self.host_in, non_blocking=True)
            self._forward()
            self.host_out.copy_(self.dev_out[0], non_blocking=True)
        torch.cuda.synchronize(self.device)

    def stage(self, images: np.ndarray, image_mask: np.ndarray, tokens: np.ndarray, token_mask: np.ndarray,
              noise: np.ndarray) -> None:  # fmt: skip
        if images.dtype != np.uint8 or images.shape != (self.num_images, _IMAGE_SIZE, _IMAGE_SIZE, 3):
            raise ValueError(f"expected uint8 images {(self.num_images, _IMAGE_SIZE, _IMAGE_SIZE, 3)}, got "
                             f"{images.dtype} {images.shape}")  # fmt: skip
        if np.size(noise) != self.action_horizon * self.action_dim:
            raise ValueError(f"expected noise of {self.action_horizon}x{self.action_dim}, got {np.shape(noise)}")
        buf = self.np_in
        buf[: self.noise_bytes].view(np.float32)[:] = np.asarray(noise, np.float32).reshape(-1)
        buf[self.tokens_at : self.masks_at].view(np.int32)[:] = tokens
        buf[self.masks_at : self.masks_at + self.max_token_len] = token_mask
        buf[self.masks_at + self.max_token_len : self.masks_at + self.max_token_len + self.num_images] = image_mask
        buf[self.images_at :].reshape(images.shape)[:] = images

    def run(self) -> np.ndarray:
        if self._graph is not None:
            self._graph.replay()
        else:
            self.dev_in.copy_(self.host_in, non_blocking=True)
            self._forward()
            self.host_out.copy_(self.dev_out[0], non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        return self.np_out.copy()


class FastPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        graph: Pi05Graph,
        *,
        image_keys: Sequence[str],
        transforms: Sequence[_transforms.DataTransformFn],
        output_transforms: Sequence[_transforms.DataTransformFn],
        metadata: Mapping[str, object],
        seed: int = 0,
    ) -> None:
        if len(image_keys) != graph.num_images:
            raise ValueError(f"{len(image_keys)} image keys for a graph of {graph.num_images} images")
        self._graph = graph
        self._image_keys = tuple(image_keys)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._metadata = metadata
        self._rng = np.random.default_rng(seed)

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
        if noise is None:
            noise = self._rng.standard_normal((self._graph.action_horizon, self._graph.action_dim), dtype=np.float32)
        self._graph.stage(
            np.stack([inputs["image"][k] for k in self._image_keys]),
            np.array([inputs["image_mask"][k] for k in self._image_keys], dtype=bool),
            inputs["tokenized_prompt"],
            inputs["tokenized_prompt_mask"],
            noise,
        )
        start = time.monotonic()
        actions = self._graph.run()
        model_time = time.monotonic() - start
        outputs = self._output_transform({"state": inputs["state"], "actions": actions})
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    @property
    def metadata(self) -> Mapping[str, object]:
        return self._metadata


def policy_transforms(
    train_config: _config.TrainConfig, checkpoint_dir: pathlib.Path, default_prompt: str | None
) -> tuple[list[_transforms.DataTransformFn], list[_transforms.DataTransformFn]]:
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(str(checkpoint_dir / "assets"), data_config.asset_id)
    quantiles = data_config.use_quantile_norm
    return (
        [
            _transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=quantiles),
            *data_config.model_transforms.inputs,
        ],
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=quantiles),
            *data_config.data_transforms.outputs,
        ],
    )


def create_fast_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    image_keys: Sequence[str],
    default_prompt: str | None = None,
    device: str = "cuda",
    compile_mode: str | None = "default",
) -> FastPolicy:
    model = train_config.model
    if type(model) is not pi0_config.Pi0Config or model.model_type != _model.ModelType.PI05:
        raise ValueError(f"the fast path serves plain pi05 only, got {type(model).__name__} {model.model_type}")
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    weights = checkpoint_dir / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found; convert the JAX checkpoint first (finetune/convert_torch.py)")
    transforms, output_transforms = policy_transforms(train_config, checkpoint_dir, default_prompt)
    graph = Pi05Graph(
        weights,
        num_images=len(image_keys),
        max_token_len=model.max_token_len,
        action_horizon=model.action_horizon,
        action_dim=model.action_dim,
        device=device,
        compile_mode=compile_mode,
    )
    return FastPolicy(
        graph,
        image_keys=image_keys,
        transforms=transforms,
        output_transforms=output_transforms,
        metadata=train_config.policy_metadata or {},
    )
