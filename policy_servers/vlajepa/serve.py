"""Bench-side VLA-JEPA policy server."""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import pathlib
import re
import sys
import traceback
from typing import Annotated, TypedDict

import msgpack
import numpy as np
import torch
import websockets.asyncio.server as _server
import websockets.frames
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel, BeforeValidator, ConfigDict

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "finetune"))

import history  # noqa: E402
import speed_conditioning  # noqa: E402
from bench_data import EXPECTED_SCHEMA, IMAGE_SIZE, unnormalize_actions  # noqa: E402
from train import ACTION_STATS, CAMERA_STAMP  # noqa: E402


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)

unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def _ndarray(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"expected an ndarray, got {type(value).__name__}")
    return value


Array = Annotated[np.ndarray, BeforeValidator(_ndarray)]


class InferRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore", populate_by_name=True)
    exterior: Array
    wrist: Array
    joint_position: Array
    gripper_position: Array
    prompt: str
    belt_speed: float | None = None

    @classmethod
    def from_wire(cls, obs: dict, num_exterior: int, *, conditioned: bool = False) -> InferRequest:
        speed = obs.get(speed_conditioning.REQUEST_FIELD)
        if (speed is not None) != conditioned:
            raise ValueError(
                f"this checkpoint was trained {'with' if conditioned else 'without'} the belt speed, so a request "
                f"must {'carry' if conditioned else 'not carry'} {speed_conditioning.REQUEST_FIELD!r}"
            )
        if speed is not None and (isinstance(speed, bool) or not isinstance(speed, (float, int, np.floating))):
            raise ValueError(f"{speed_conditioning.REQUEST_FIELD}: expected a float in m/s, got {speed!r}")
        req = cls(
            exterior=obs.get("observation/exterior_image_1_left"),
            wrist=obs.get("observation/wrist_image_left"),
            joint_position=obs.get("observation/joint_position"),
            gripper_position=obs.get("observation/gripper_position"),
            prompt=obs.get("prompt"),
            belt_speed=None if speed is None else float(speed),
        )
        frame = (IMAGE_SIZE, IMAGE_SIZE, 3)
        exterior_shape = (num_exterior, *frame) if num_exterior > 1 else frame
        for name, value, shape, dtype in (
            ("exterior_image_1_left", req.exterior, exterior_shape, np.uint8),
            ("wrist_image_left", req.wrist, frame, np.uint8),
            ("joint_position", req.joint_position, (7,), np.float32),
            ("gripper_position", req.gripper_position, (1,), np.float32),
        ):
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"{name}: expected {np.dtype(dtype).name}{shape}, got {value.dtype.name}{value.shape}")
        return req


class InferResponse(TypedDict):
    actions: np.ndarray


class ConnectMetadata(TypedDict):
    model: str
    exterior_history: list[int]
    bench_schema: str
    action_horizon: int
    speed_conditioned: bool


def latest_checkpoint(run_dir: pathlib.Path) -> pathlib.Path:
    steps = {
        int(m.group(1)): p
        for p in (run_dir / "checkpoints").glob("step_*")
        if (m := re.fullmatch(r"step_(\d+)", p.name))
    }
    if not steps:
        raise FileNotFoundError(f"no checkpoints/step_* under {run_dir}")
    return steps[max(steps)]


def check_camera_stamp(run_dir: pathlib.Path) -> str:
    declared = json.loads((run_dir / CAMERA_STAMP).read_text()).get("bench_schema")
    if declared != EXPECTED_SCHEMA:
        raise RuntimeError(
            f"{run_dir} was trained on camera schema {declared!r}; the bench client renders {EXPECTED_SCHEMA!r}"
        )
    return declared


class CheckpointPolicy:
    def __init__(
        self,
        run_dir: pathlib.Path,
        ckpt: pathlib.Path,
        qwen: pathlib.Path,
        vjepa: pathlib.Path,
        attn: str,
        seed: int,
        speed_norm: speed_conditioning.Normalisation | None,
    ) -> None:
        from starVLA.model.framework import build_framework

        cfg = OmegaConf.load(run_dir / "config.yaml")
        cfg.framework.qwenvl.base_vlm = str(qwen)
        cfg.framework.vj2_model.base_encoder = str(vjepa)
        cfg.framework.qwenvl.attn_implementation = attn
        stats = json.loads((run_dir / ACTION_STATS).read_text())
        self.action_min = np.asarray(stats["action_min"], dtype=np.float32)
        self.action_max = np.asarray(stats["action_max"], dtype=np.float32)
        self.history = history.offsets_from_env()
        self.horizon = int(cfg.framework.action_model.action_horizon)
        self.seed = seed
        self.speed_norm = speed_norm
        self.model = build_framework(cfg)
        self.model.load_state_dict(torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True), strict=True)
        self.model = self.model.cuda().eval()

    def infer(self, obs: dict) -> InferResponse:
        req = InferRequest.from_wire(obs, len(self.history) + 1, conditioned=self.speed_norm is not None)
        exteriors = req.exterior if req.exterior.ndim == 4 else req.exterior[None]
        images = [Image.fromarray(np.ascontiguousarray(f)) for f in exteriors]
        images.append(Image.fromarray(np.ascontiguousarray(req.wrist)))
        state = np.concatenate([req.joint_position, req.gripper_position])[None].astype(np.float32)
        if self.speed_norm is not None:
            assert req.belt_speed is not None
            speed = speed_conditioning.normalize(req.belt_speed, self.speed_norm)
            state = np.concatenate([state, [[speed]]], axis=1, dtype=np.float32)
        torch.manual_seed(self.seed)
        out = self.model.predict_action(batch_images=[images], instructions=[req.prompt], state=[state])
        actions = unnormalize_actions(np.asarray(out["normalized_actions"][0]), self.action_min, self.action_max)
        actions[:, 7] = np.clip(actions[:, 7], 0.0, 1.0)
        return {"actions": actions}

    def metadata(self) -> ConnectMetadata:
        return {
            "model": "vla_jepa",
            history.METADATA_KEY: list(self.history),
            "bench_schema": EXPECTED_SCHEMA,
            "action_horizon": self.horizon,
            speed_conditioning.METADATA_KEY: self.speed_norm is not None,
        }


async def _handler(policy: CheckpointPolicy, websocket: _server.ServerConnection) -> None:
    logging.info("Connection from %s opened", websocket.remote_address)
    packer = Packer()
    await websocket.send(packer.pack(policy.metadata()))
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


async def _serve(policy: CheckpointPolicy, port: int) -> None:
    async with _server.serve(
        functools.partial(_handler, policy), "0.0.0.0", port, compression=None, max_size=None
    ) as server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None, help="step dir; default: the latest")
    parser.add_argument("--qwen", required=True, type=pathlib.Path)
    parser.add_argument("--vjepa", required=True, type=pathlib.Path)
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--seed", type=int, default=0, help="flow-matching noise seed, reset on every request")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--speed-conditioned", action="store_true", help="the checkpoint takes the belt speed in its state"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    run_dir = args.run_dir.expanduser().resolve()
    ckpt = args.checkpoint.expanduser().resolve() if args.checkpoint else latest_checkpoint(run_dir)
    history.check_serve(ckpt, history.offsets_from_env())
    check_camera_stamp(run_dir)
    state_dim = int(OmegaConf.load(run_dir / "config.yaml").framework.action_model.state_dim)
    speed_norm = speed_conditioning.check_serve(run_dir, args.speed_conditioned, state_dim)
    logging.info("Loading %s", ckpt)
    policy = CheckpointPolicy(run_dir, ckpt, args.qwen, args.vjepa, args.attn, args.seed, speed_norm)
    logging.info("Serving on 0.0.0.0:%d with metadata %s", args.port, policy.metadata())
    asyncio.run(_serve(policy, args.port))


if __name__ == "__main__":
    main()
