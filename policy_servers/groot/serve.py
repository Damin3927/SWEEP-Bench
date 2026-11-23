"""Bench-side GR00T N1.7 policy server."""

from __future__ import annotations

import argparse
import asyncio
import functools
import importlib.util
import json
import logging
import pathlib
import sys
import traceback
from typing import TYPE_CHECKING, Annotated, TypedDict

import msgpack
import numpy as np
import torch
import websockets.asyncio.server as _server
import websockets.frames
from pydantic import BaseModel, BeforeValidator, ConfigDict

if TYPE_CHECKING:
    import types

_FINETUNE = pathlib.Path(__file__).resolve().parent / "finetune"


def _load(name: str) -> types.ModuleType:
    module_name = f"groot_bench_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _FINETUNE / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to locate {name} under {_FINETUNE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _pack_array(obj: object) -> object:
    if isinstance(obj, np.ndarray | np.generic) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict[bytes, object]) -> object:
    if b"__ndarray__" in obj:
        data, dtype, shape = obj[b"data"], obj[b"dtype"], obj[b"shape"]
        if not isinstance(data, bytes) or not isinstance(dtype, str) or not isinstance(shape, list | tuple):
            raise ValueError("malformed ndarray in request")
        dims = tuple(n for n in shape if isinstance(n, int))
        if len(dims) != len(shape):
            raise ValueError("malformed ndarray shape in request")
        return np.ndarray(buffer=data, dtype=np.dtype(dtype), shape=dims)
    if b"__npgeneric__" in obj:
        dtype = obj[b"dtype"]
        if not isinstance(dtype, str):
            raise ValueError("malformed numpy scalar in request")
        return np.dtype(dtype).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)

unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def _array(dtype: type[np.generic], ndims: tuple[int, ...], last: int | None = None) -> BeforeValidator:
    def check(value: object) -> np.ndarray:
        if not isinstance(value, np.ndarray) or value.dtype != dtype or value.ndim not in ndims:
            got = f"{value.dtype} {value.shape}" if isinstance(value, np.ndarray) else type(value).__name__
            raise ValueError(f"expected a {np.dtype(dtype)} array with ndim in {ndims}, got {got}")
        if last is not None and value.shape[-1] != last:
            raise ValueError(f"expected trailing dimension {last}, got shape {value.shape}")
        return value

    return BeforeValidator(check)


class InferRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    exterior: Annotated[np.ndarray, _array(np.uint8, (3, 4), last=3)]
    wrist: Annotated[np.ndarray, _array(np.uint8, (3,), last=3)]
    joint_position: Annotated[np.ndarray, _array(np.float32, (1,), last=7)]
    gripper_position: Annotated[np.ndarray, _array(np.float32, (1,), last=1)]
    prompt: str

    @classmethod
    def from_wire(cls, raw: object) -> InferRequest:
        if not isinstance(raw, dict):
            raise ValueError(f"expected a request dict, got {type(raw).__name__}")
        prefix = "observation/"
        fields = {
            "exterior_image_1_left": "exterior",
            "wrist_image_left": "wrist",
            "joint_position": "joint_position",
            "gripper_position": "gripper_position",
        }
        renamed = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                raise ValueError(f"request keys must be strings, got {key!r}")
            renamed[fields.get(key.removeprefix(prefix), key) if key.startswith(prefix) else key] = value
        return cls.model_validate(renamed)


class InferResponse(TypedDict):
    actions: np.ndarray


class ConnectMetadata(TypedDict):
    exterior_history: list[int]
    bench_schema: str
    action_horizon: int


def check_attention_backend(checkpoint: pathlib.Path) -> None:
    config = json.loads((checkpoint / "config.json").read_text())
    if not config.get("use_flash_attention", False):
        return
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"{checkpoint} was trained with flash-attn, but flash_attn does not import here ({exc}); "
            "serve it where the run trained, e.g. inside the run's container image"
        ) from exc


class CheckpointPolicy:
    def __init__(self, checkpoint: pathlib.Path, exterior_history: tuple[int, ...], device: str) -> None:
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.bench = _load("bench_embodiment")
        self.exterior_history = exterior_history
        self.policy = Gr00tPolicy(self.bench.embodiment_tag(), str(checkpoint), device=device, strict=True)
        expected = self.bench.video_keys(exterior_history)
        served = list(self.policy.modality_configs["video"].modality_keys)
        if served != expected:
            raise RuntimeError(
                f"{checkpoint}'s processor reads video keys {served}, but exterior history "
                f"{exterior_history} needs {expected}"
            )
        self.state_keys = list(self.policy.modality_configs["state"].modality_keys)
        self.action_keys = list(self.policy.modality_configs["action"].modality_keys)
        self.action_horizon = len(self.policy.modality_configs["action"].delta_indices)
        if self.state_keys != list(self.bench.STATE_KEYS) or self.action_keys != list(self.bench.STATE_KEYS):
            raise RuntimeError(f"unexpected state/action keys {self.state_keys} / {self.action_keys}")

    def infer(self, raw: object) -> InferResponse:
        request = InferRequest.from_wire(raw)
        observation = self.bench.groot_observation(
            request.exterior,
            request.wrist,
            request.joint_position,
            request.gripper_position,
            request.prompt,
            self.exterior_history,
        )
        action, _ = self.policy.get_action(observation)
        chunk = np.concatenate([action[k][0] for k in self.action_keys], axis=-1).astype(np.float32)
        return {"actions": chunk}


async def _handler(policy: CheckpointPolicy, metadata: ConnectMetadata, websocket: _server.ServerConnection) -> None:
    logging.info("Connection from %s opened", websocket.remote_address)
    packer = Packer()
    await websocket.send(packer.pack(metadata))
    while True:
        try:
            obs = unpackb(await websocket.recv())
            await websocket.send(packer.pack(policy.infer(obs)))
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


async def _serve(policy: CheckpointPolicy, metadata: ConnectMetadata, host: str, port: int) -> None:
    async with _server.serve(
        functools.partial(_handler, policy, metadata), host, port, compression=None, max_size=None
    ) as server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="a run's checkpoint-<step> dir")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0, help="torch seed set once at start (flow-matching noise)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    history = _load("history")
    camera_vintage = _load("prep_dataset").load_camera_vintage()
    checkpoint = args.checkpoint.expanduser().resolve()
    offsets = history.offsets_from_env()
    history.check_serve(checkpoint, offsets)
    camera_vintage.check_checkpoint_camera_vintage(checkpoint)
    declared, _, _ = camera_vintage.read_checkpoint_vintage(checkpoint)
    if declared is None:
        raise RuntimeError(f"{checkpoint} declares no camera vintage")
    check_attention_backend(checkpoint)

    _load("hf_offline").install()
    torch.manual_seed(args.seed)
    logging.info("Loading %s (exterior history %s)", checkpoint, offsets)
    policy = CheckpointPolicy(checkpoint, offsets, args.device)
    metadata: ConnectMetadata = {
        "exterior_history": list(offsets),
        "bench_schema": declared,
        "action_horizon": policy.action_horizon,
    }
    logging.info("Serving on 0.0.0.0:%d with metadata %s", args.port, metadata)
    asyncio.run(_serve(policy, metadata, "0.0.0.0", args.port))


if __name__ == "__main__":
    main()
