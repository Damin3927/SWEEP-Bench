"""Bench-side Cosmos Policy server."""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import pathlib
import pickle
import sys
import time
import traceback
from typing import TYPE_CHECKING, Annotated, TypedDict

import msgpack
import numpy as np
import pydantic
import torch
import websockets.asyncio.server as _server
import websockets.frames

if TYPE_CHECKING:
    from collections.abc import Mapping

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from finetune.bench_config import CHUNK_SIZE, INFERENCE_EXPERIMENT_NAME  # noqa: E402
from finetune.bench_dataset import T5_FILE, load_stats, rescale  # noqa: E402
from finetune.history import LatentLayout, check_serve, offsets_from_env  # noqa: E402

IMAGE_SIZE = 224
ACTION_DIM = 8


class InferResponse(TypedDict):
    actions: np.ndarray


class ConnectMetadata(TypedDict):
    exterior_history: list[int]
    bench_schema: str | None
    action_horizon: int


def _pack_array(obj: object) -> object:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict[bytes, object]) -> object:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)

unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def _array(dtype: type, ndim: tuple[int, ...]):
    def check(value: object) -> np.ndarray:
        if not isinstance(value, np.ndarray) or value.dtype != dtype or value.ndim not in ndim:
            raise ValueError(f"expected a {dtype.__name__} array with ndim in {ndim}, got {type(value).__name__}")
        return value

    return pydantic.BeforeValidator(check)


class InferRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    exterior: Annotated[np.ndarray, _array(np.uint8, (3, 4))] = pydantic.Field(
        alias="observation/exterior_image_1_left"
    )
    wrist: Annotated[np.ndarray, _array(np.uint8, (3,))] = pydantic.Field(alias="observation/wrist_image_left")
    joint_position: Annotated[np.ndarray, _array(np.float32, (1,))] = pydantic.Field(alias="observation/joint_position")
    gripper_position: Annotated[np.ndarray, _array(np.float32, (1,))] = pydantic.Field(
        alias="observation/gripper_position"
    )
    prompt: str

    def exterior_frames(self, expected: int) -> np.ndarray:
        frames = self.exterior if self.exterior.ndim == 4 else self.exterior[None]
        if frames.shape[0] != expected or frames.shape[-1] != 3:
            raise ValueError(f"exterior: expected ({expected}, H, W, 3) for this checkpoint, got {frames.shape}")
        return frames

    def proprio(self) -> np.ndarray:
        if self.joint_position.shape != (7,) or self.gripper_position.shape != (1,):
            raise ValueError(f"proprio: got {self.joint_position.shape} + {self.gripper_position.shape}, want 7 + 1")
        return np.concatenate([self.joint_position, self.gripper_position])


def build_video(exterior: list[np.ndarray], wrist: np.ndarray, layout: LatentLayout) -> np.ndarray:
    sources = {
        "blank": np.zeros_like(wrist),
        "wrist": wrist,
        "future_wrist": wrist,
        "future_exterior": exterior[-1],
        **{f"exterior@{o}": e for o, e in zip((*layout.history, 0), exterior, strict=True)},
    }
    return np.stack([sources[s] for s in layout.frame_sources()])


def load_dcp_model(model: torch.nn.Module, model_dir: pathlib.Path) -> None:
    from cosmos_policy._src.predict2.checkpointer.dcp import ModelWrapper
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint import load as dcp_load

    wrapper = ModelWrapper(model)
    state = wrapper.state_dict()
    saved = set(FileSystemReader(str(model_dir)).read_metadata().state_dict_metadata)
    missing = sorted(k for k in state if k not in saved and k.startswith("net."))
    if missing:
        raise RuntimeError(f"{model_dir} lacks {len(missing)} model weights, e.g. {missing[:5]}")
    state = {k: v for k, v in state.items() if k in saved}
    dcp_load(state, storage_reader=FileSystemReader(str(model_dir)))
    wrapper.load_state_dict(state)
    logging.info("loaded %d tensors from %s", len(state), model_dir)


class CosmosPolicyServer:
    def __init__(
        self,
        checkpoint_dir: pathlib.Path,
        assets_dir: pathlib.Path,
        num_denoising_steps: int,
        seed: int,
        t5_live: bool,
    ) -> None:
        from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

        self.history, self.bench_schema = check_serve(checkpoint_dir, offsets_from_env())
        self.layout = LatentLayout(self.history)
        self.stats = load_stats(assets_dir)
        with open(assets_dir / T5_FILE, "rb") as f:
            self.t5_cache: dict[str, torch.Tensor] = pickle.load(f)
        self.t5_live = t5_live
        if t5_live:
            prompt = min(self.t5_cache)
            torch.testing.assert_close(self._encode(prompt), self.t5_cache[prompt], rtol=1.6e-2, atol=1e-5)
        self.num_denoising_steps = num_denoising_steps
        self.seed = seed
        self.model, self.config = load_model_from_checkpoint(
            experiment_name=INFERENCE_EXPERIMENT_NAME,
            s3_checkpoint_dir="",
            config_file="finetune/bench_config.py",
            load_ema_to_reg=False,
            skip_load_model=True,
        )
        load_dcp_model(self.model, checkpoint_dir / "model" if (checkpoint_dir / "model").is_dir() else checkpoint_dir)
        if self.model.config.state_t != self.layout.state_t:
            raise RuntimeError(f"model state_t {self.model.config.state_t} != layout {self.layout}")
        self.model.eval()
        self.model.to("cuda")

    def metadata(self) -> ConnectMetadata:
        return {"exterior_history": list(self.history), "bench_schema": self.bench_schema, "action_horizon": CHUNK_SIZE}

    def _text_embedding(self, prompt: str) -> torch.Tensor:
        cached = self.t5_cache.get(prompt)
        if cached is not None:
            return cached
        if not self.t5_live:
            raise KeyError(f"prompt {prompt!r} is not among the {len(self.t5_cache)} trained prompts; pass --t5-live")
        logging.warning("encoding unseen prompt live with T5-11B: %r", prompt)
        embedding = self._encode(prompt)
        self.t5_cache[prompt] = embedding
        return embedding

    @staticmethod
    def _encode(prompt: str) -> torch.Tensor:
        from cosmos_policy._src.predict2.inference.get_t5_emb import get_text_embedding

        return get_text_embedding(prompt).to(dtype=torch.bfloat16).cpu()

    @torch.inference_mode()
    def infer(self, raw: Mapping[str, object]) -> InferResponse:
        from cosmos_policy.experiments.robot.cosmos_utils import (
            apply_image_transforms,
            extract_action_chunk_from_latent_sequence,
        )

        request = InferRequest.model_validate(raw)
        exterior = request.exterior_frames(len(self.history) + 1)
        images = apply_image_transforms(np.concatenate([exterior, request.wrist[None]], axis=0))
        if images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(f"images must be {IMAGE_SIZE}x{IMAGE_SIZE} (the dataset's size), got {images.shape}")
        video = build_video(list(images[:-1]), images[-1], self.layout)
        proprio = rescale(request.proprio(), self.stats, "proprio")

        layout = self.layout

        def index(i: int) -> torch.Tensor:
            return torch.tensor([i], dtype=torch.int64, device="cuda")

        batch = {
            "dataset_name": "video_data",
            "video": torch.from_numpy(np.ascontiguousarray(video.transpose(3, 0, 1, 2))[None]).cuda(),
            "t5_text_embeddings": self._text_embedding(request.prompt).reshape(1, 512, -1).to("cuda", torch.bfloat16),
            "fps": torch.tensor([16], dtype=torch.bfloat16, device="cuda"),
            "padding_mask": torch.zeros((1, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.bfloat16, device="cuda"),
            "num_conditional_frames": layout.num_conditional_frames,
            "proprio": torch.from_numpy(proprio).reshape(1, -1).to("cuda", torch.bfloat16),
            "current_proprio_latent_idx": index(layout.current_proprio),
            "current_wrist_image_latent_idx": index(layout.current_wrist),
            "current_wrist_image2_latent_idx": index(-1),
            "current_image_latent_idx": index(layout.current_exterior),
            "current_image2_latent_idx": index(-1),
            "action_latent_idx": index(layout.action),
            "future_proprio_latent_idx": index(layout.future_proprio),
            "future_wrist_image_latent_idx": index(layout.future_wrist),
            "future_wrist_image2_latent_idx": index(-1),
            "future_image_latent_idx": index(layout.future_exterior),
            "future_image2_latent_idx": index(-1),
            "value_latent_idx": index(layout.value),
        }
        latent = self.model.generate_samples_from_batch(
            batch,
            n_sample=1,
            num_steps=self.num_denoising_steps,
            seed=self.seed,
            is_negative_prompt=False,
            use_variance_scale=False,
        )
        normalized = extract_action_chunk_from_latent_sequence(
            latent, action_shape=(CHUNK_SIZE, ACTION_DIM), action_indices=index(layout.action)
        )[0]
        lo, hi = self.stats["actions_min"], self.stats["actions_max"]
        actions = 0.5 * (normalized.float().cpu().numpy() + 1) * (hi - lo) + lo
        return {"actions": actions.astype(np.float32)}


async def _handler(policy: CosmosPolicyServer, websocket: _server.ServerConnection) -> None:
    logging.info("Connection from %s opened", websocket.remote_address)
    packer = Packer()
    await websocket.send(packer.pack(policy.metadata()))
    while True:
        try:
            obs = unpackb(await websocket.recv())
            start = time.perf_counter()
            response = policy.infer(obs)
            logging.info(
                "infer %.0f ms, exterior %s",
                1000 * (time.perf_counter() - start),
                getattr(obs.get("observation/exterior_image_1_left"), "shape", None),
            )
            await websocket.send(packer.pack(response))
        except websockets.ConnectionClosed:
            logging.info("Connection from %s closed", websocket.remote_address)
            break
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Internal server error. Traceback included in previous frame.",
            )
            raise


async def _serve(policy: CosmosPolicyServer, host: str, port: int) -> None:
    async with _server.serve(
        functools.partial(_handler, policy), host, port, compression=None, max_size=None
    ) as server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True, help="<job>/checkpoints/iter_*")
    parser.add_argument("--assets-dir", type=pathlib.Path, required=True, help="the run's stats + T5 cache")
    parser.add_argument("--num-denoising-steps", type=int, default=5, help="upstream's LIBERO/RoboCasa eval value")
    parser.add_argument("--seed", type=int, default=195, help="sampling seed used on every query (upstream: 195)")
    parser.add_argument("--t5-live", action="store_true", help="encode prompts missing from the cache with T5-11B")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    policy = CosmosPolicyServer(
        args.checkpoint_dir.resolve(), args.assets_dir.resolve(), args.num_denoising_steps, args.seed, args.t5_live
    )
    logging.info("Serving on 0.0.0.0:%d with metadata %s", args.port, policy.metadata())
    asyncio.run(_serve(policy, "0.0.0.0", args.port))


if __name__ == "__main__":
    main()
