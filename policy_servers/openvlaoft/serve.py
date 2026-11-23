"""Bench-side OpenVLA-OFT policy server."""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import pathlib
import sys
import time
import traceback
from typing import TYPE_CHECKING, Annotated, TypedDict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "finetune"))

import bench_common as bc  # noqa: E402  (must precede every prismatic import)
import msgpack  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import websockets.asyncio.server as _server  # noqa: E402
import websockets.frames  # noqa: E402
from peft import PeftModel  # noqa: E402
from PIL import Image  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor  # noqa: E402
from prismatic.models.action_heads import L1RegressionActionHead  # noqa: E402
from prismatic.models.projectors import ProprioProjector  # noqa: E402
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field  # noqa: E402
from train import CHECKPOINTS, STEP_DIR, image_processor, register_auto_classes  # noqa: E402

if TYPE_CHECKING:
    from transformers import BatchFeature


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


def _array(dtype: str, shape: tuple[int | None, ...]):
    def check(value: object) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"expected an ndarray, got {type(value).__name__}")
        if value.dtype != np.dtype(dtype):
            raise ValueError(f"expected dtype {dtype}, got {value.dtype}")
        if value.ndim != len(shape) or any(s is not None and s != v for s, v in zip(shape, value.shape, strict=True)):
            raise ValueError(f"expected shape {shape}, got {value.shape}")
        return value

    return Annotated[np.ndarray, BeforeValidator(check)]


Image224 = _array("uint8", (bc.IMAGE_SIZE, bc.IMAGE_SIZE, 3))
JointPosition = _array("float32", (7,))
GripperPosition = _array("float32", (1,))


class DroidRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore", frozen=True)

    exterior: np.ndarray = Field(alias="observation/exterior_image_1_left")
    wrist: Image224 = Field(alias="observation/wrist_image_left")
    joint_position: JointPosition = Field(alias="observation/joint_position")
    gripper_position: GripperPosition = Field(alias="observation/gripper_position")
    prompt: str = Field(min_length=1)


class InferResponse(TypedDict):
    actions: np.ndarray


class ConnectMetadata(TypedDict):
    exterior_history: list[int]
    action_horizon: int
    bench_schema: str


def exterior_frames(request: DroidRequest, offsets: tuple[int, ...]) -> list[np.ndarray]:
    expected = (len(offsets) + 1, bc.IMAGE_SIZE, bc.IMAGE_SIZE, 3) if offsets else (bc.IMAGE_SIZE, bc.IMAGE_SIZE, 3)
    ext = request.exterior
    if ext.dtype != np.uint8 or ext.shape != expected:
        raise ValueError(
            f"exterior history {offsets} needs observation/exterior_image_1_left uint8 {expected}, "
            f"got {ext.dtype} {ext.shape}"
        )
    return list(ext) if offsets else [ext]


def model_images(request: DroidRequest, offsets: tuple[int, ...], crop: bool) -> list[np.ndarray]:
    *past, current = exterior_frames(request, offsets)
    images = [current, request.wrist, *past]
    return [bc.center_crop(img) for img in images] if crop else images


def build_inputs(
    processor: PrismaticProcessor,
    stats: bc.DatasetStats,
    request: DroidRequest,
    offsets: tuple[int, ...],
    crop: bool,
) -> tuple[BatchFeature, np.ndarray]:
    primary, *others = (Image.fromarray(img) for img in model_images(request, offsets, crop))
    inputs = processor(bc.prompt_text(request.prompt), primary)
    extra = [image_processor(processor).apply_transform(img)[None] for img in others]
    inputs["pixel_values"] = torch.cat([inputs["pixel_values"], *extra], dim=1)
    state = np.concatenate([request.joint_position, request.gripper_position])
    return inputs, bc.normalize(state, stats["proprio"])


def latest_step(run_dir: pathlib.Path) -> pathlib.Path:
    steps = {int(m.group(1)): p for p in (run_dir / CHECKPOINTS).iterdir() if (m := STEP_DIR.fullmatch(p.name))}
    if not steps:
        raise FileNotFoundError(f"no {CHECKPOINTS}/step_* under {run_dir}")
    return steps[max(steps)]


class CheckpointPolicy:
    def __init__(self, run_dir: pathlib.Path, step_dir: pathlib.Path, base: str, offsets: tuple[int, ...]) -> None:
        cfg = json.loads((run_dir / bc.RUN_CONFIG_FILE).read_text())
        if tuple(cfg["exterior_history"]) != offsets or cfg["num_images"] != bc.num_images(offsets):
            raise RuntimeError(f"{run_dir}/{bc.RUN_CONFIG_FILE} disagrees with exterior history {offsets}: {cfg}")
        for key, value in (
            ("num_actions_chunk", bc.NUM_ACTIONS_CHUNK),
            ("action_dim", bc.ACTION_DIM),
            ("proprio_dim", bc.PROPRIO_DIM),
        ):
            if cfg[key] != value:
                raise RuntimeError(f"{run_dir} trained {key}={cfg[key]}, this tree builds {value}")
        self.offsets = offsets
        self.crop = bool(cfg["image_aug"])
        self.device = torch.device("cuda")
        register_auto_classes()
        self.processor = PrismaticProcessor.from_pretrained(base)
        vla = OpenVLAForActionPrediction.from_pretrained(base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        vla.vision_backbone.set_num_images_in_input(bc.num_images(offsets))
        vla = PeftModel.from_pretrained(vla.to(self.device), step_dir / "lora_adapter").merge_and_unload()
        self.stats = bc.read_stats(step_dir / bc.STATS_FILE)
        vla.norm_stats = {**vla.norm_stats, bc.UNNORM_KEY: self.stats}
        self.vla = vla.eval()
        llm_dim = vla.llm_dim
        self.proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=bc.PROPRIO_DIM)
        self.proprio_projector.load_state_dict(torch.load(step_dir / "proprio_projector.pt", map_location="cpu"))
        self.proprio_projector = self.proprio_projector.to(torch.bfloat16).to(self.device).eval()
        self.action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=bc.ACTION_DIM)
        self.action_head.load_state_dict(torch.load(step_dir / "action_head.pt", map_location="cpu"))
        self.action_head = self.action_head.to(torch.bfloat16).to(self.device).eval()

    @torch.inference_mode()
    def infer(self, raw: object) -> InferResponse:
        request = DroidRequest.model_validate(raw)
        inputs, proprio = build_inputs(self.processor, self.stats, request, self.offsets, self.crop)
        inputs = inputs.to(self.device, dtype=torch.bfloat16)
        actions, _ = self.vla.predict_action(
            **inputs,
            unnorm_key=bc.UNNORM_KEY,
            do_sample=False,
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            action_head=self.action_head,
            use_film=False,
        )
        return {"actions": bc.to_absolute(np.asarray(actions, dtype=np.float32), request.joint_position)}


async def _handler(policy: CheckpointPolicy, metadata: ConnectMetadata, websocket: _server.ServerConnection) -> None:
    logging.info("Connection from %s opened", websocket.remote_address)
    packer = Packer()
    await websocket.send(packer.pack(metadata))
    while True:
        try:
            obs = unpackb(await websocket.recv())
            t0 = time.perf_counter()
            response = policy.infer(obs)
            logging.info("infer %.1f ms", 1000 * (time.perf_counter() - t0))
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


async def _serve(policy: CheckpointPolicy, metadata: ConnectMetadata, port: int) -> None:
    handler = functools.partial(_handler, policy, metadata)
    async with _server.serve(handler, "0.0.0.0", port, compression=None, max_size=None) as server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=pathlib.Path, required=True, help="finetune/train.py --run-dir")
    parser.add_argument("--step-dir", type=pathlib.Path, default=None, help="default: the run's latest step_*")
    parser.add_argument("--base", required=True, help="openvla/openvla-7b snapshot dir the run fine-tuned")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)

    run_dir = args.run_dir.resolve()
    step_dir = (args.step_dir or latest_step(run_dir)).resolve()
    if run_dir not in step_dir.parents:
        raise ValueError(f"{step_dir} is not inside {run_dir}")
    offsets = bc.history.offsets_from_env()
    bc.history.check_serve(step_dir, offsets)
    logging.info("Loading %s (base %s, exterior history %s)", step_dir, args.base, offsets)
    policy = CheckpointPolicy(run_dir, step_dir, args.base, offsets)
    if bc.history.METADATA_KEY != "exterior_history":
        raise RuntimeError(f"history.METADATA_KEY is {bc.history.METADATA_KEY!r}; update ConnectMetadata")
    metadata: ConnectMetadata = {
        "exterior_history": list(offsets),
        "action_horizon": bc.NUM_ACTIONS_CHUNK,
        "bench_schema": bc.DATASET_SCHEMA,
    }
    logging.info("Serving on 0.0.0.0:%d, metadata %s", args.port, metadata)
    asyncio.run(_serve(policy, metadata, args.port))


if __name__ == "__main__":
    main()
