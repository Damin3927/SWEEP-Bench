"""Bench-side FastWAM policy server."""

import argparse
import asyncio
import functools
import hashlib
import importlib.util
import json
import logging
import os
import pathlib
import re
import sys
import traceback
from collections.abc import Callable, Mapping
from typing import TypeAlias, TypedDict

import msgpack
import numpy as np
import torch
import websockets.asyncio.server as _server
import websockets.frames


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
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

_fp_spec = importlib.util.spec_from_file_location(
    "bench_frame_push", pathlib.Path(__file__).resolve().parent.parent / "frame_push.py"
)
assert _fp_spec is not None and _fp_spec.loader is not None
frame_push = importlib.util.module_from_spec(_fp_spec)
_fp_spec.loader.exec_module(frame_push)

unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

TEXT_CACHE_SUFFIX = "wan22ti2v5b"

SCHEMA_INFO_KEY = "bench_schema"
EULER_STAMP = "bench_euler_convention.json"

InferRequest: TypeAlias = Mapping[str, np.ndarray | str]  # noqa: UP040


class ActionResult(TypedDict):
    action: torch.Tensor


class InferResponse(TypedDict):
    actions: np.ndarray


class ConnectMetadata(TypedDict):
    obs_delta_indices: list[int]
    n_action_steps: int
    euler_convention: str
    bench_schema: str
    speed_conditioned: bool
    fast_b: None


def history_offsets(num_history_frames: int, history_stride_env_steps: int) -> list[int]:
    return [-(num_history_frames - i) * history_stride_env_steps for i in range(num_history_frames)] + [0]


CAMERA_KEYS = ("opst_cam", "wrist_cam")


def history_camera_indices(history_camera_keys: list[str] | None, run_camera_keys: list[str]) -> list[int] | None:
    if history_camera_keys is None:
        return None
    if tuple(run_camera_keys) != CAMERA_KEYS:
        raise ValueError(f"run cameras {run_camera_keys} differ from the served concat order {list(CAMERA_KEYS)}")
    unknown = [key for key in history_camera_keys if key not in CAMERA_KEYS]
    if unknown or not history_camera_keys:
        raise ValueError(f"history_camera_keys {history_camera_keys} must name cameras of {list(CAMERA_KEYS)}")
    return [CAMERA_KEYS.index(key) for key in history_camera_keys]


def history_frame_width(video_w: int, cameras: list[int] | None) -> int:
    return video_w if cameras is None else video_w // len(CAMERA_KEYS) * len(cameras)


def select_history_cameras(frames: torch.Tensor, cameras: list[int] | None) -> torch.Tensor:
    if cameras is None:
        return frames
    camera_w = frames.shape[-1] // len(CAMERA_KEYS)
    return torch.cat([frames[..., i * camera_w : (i + 1) * camera_w] for i in cameras], dim=-1)


SPEED_KEY = "belt_speed"


def run_speed_conditioned(speed_list: str | None, model_speed_conditioned: bool) -> bool:
    if model_speed_conditioned != (speed_list is not None):
        raise ValueError(
            f"run config disagrees on the belt speed: model.speed_conditioned={model_speed_conditioned}, "
            f"data.train.speed_list={speed_list!r}"
        )
    return model_speed_conditioned


def check_speed_request(speed_conditioned: bool, obs: InferRequest) -> None:
    if speed_conditioned != (SPEED_KEY in obs):
        raise ValueError(
            f"run speed_conditioned={speed_conditioned} but the request "
            f"{'carries' if SPEED_KEY in obs else 'lacks'} {SPEED_KEY!r}; serve and evaluate with matching --speed-token"
        )


def latest_weights(run_dir: pathlib.Path) -> pathlib.Path:
    by_step = {
        int(m.group(1)): p
        for p in (run_dir / "checkpoints" / "weights").glob("step_*.pt")
        if (m := re.fullmatch(r"step_(\d+)\.pt", p.name))
    }
    if not by_step:
        raise FileNotFoundError(f"no checkpoints/weights/step_*.pt under {run_dir}")
    return by_step[max(by_step)]


def dataset_declarations(dataset_root: pathlib.Path) -> tuple[str, str]:
    euler = json.loads((dataset_root / "meta" / EULER_STAMP).read_text())["euler_convention"]
    vintage = json.loads((dataset_root / "meta" / "info.json").read_text())[SCHEMA_INFO_KEY]["name"]
    return str(euler), str(vintage)


class CheckpointPolicy:
    def __init__(
        self,
        run_dir: pathlib.Path,
        weights: pathlib.Path,
        fastwam_dir: pathlib.Path,
        dataset_root: pathlib.Path | None,
        text_cache_dir: pathlib.Path | None,
        num_inference_steps: int | None,
        seed: int,
        cuda_graph: bool = False,
    ) -> None:
        sys.path.insert(0, str(fastwam_dir / "src"))
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(fastwam_dir / "checkpoints"))

        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from fastwam.models.wan22.fastwam import FastWAM
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(run_dir / "config.yaml")
        dt = cfg.data.train
        self.speed_conditioned = run_speed_conditioned(
            dt.get("speed_list"), bool(cfg.model.get("speed_conditioned", False))
        )
        if self.speed_conditioned:
            from fastwam.utils.belt_speed import encode_speed

            self._encode_speed = encode_speed
        if len(dt.dataset_dirs) != 1:
            raise ValueError(f"expected one training dataset, got {list(dt.dataset_dirs)}; pass --dataset-root")
        root = dataset_root or (fastwam_dir / str(dt.dataset_dirs[0]))
        self.euler_convention, self.bench_schema = dataset_declarations(root)
        self.video_h, self.video_w = (int(s) for s in dt.video_size)
        self.text_cache_dir = text_cache_dir or (fastwam_dir / str(dt.text_embedding_cache_dir))
        self.context_len = int(dt.context_len)
        self.history_len = int(dt.num_history_frames)
        self.history_cameras = history_camera_indices(
            None if dt.get("history_camera_keys") is None else [str(k) for k in dt.history_camera_keys],
            [str(meta.key) for meta in dt.shape_meta.images],
        )
        self.obs_delta_indices = history_offsets(
            self.history_len, int(dt.history_stride) * int(dt.global_sample_stride)
        )
        self.action_horizon = int(dt.num_frames) - 1
        self.num_inference_steps = (
            num_inference_steps if num_inference_steps is not None else int(cfg.eval_num_inference_steps)
        )
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        mc = cfg.model
        video_dit_config = OmegaConf.to_container(mc.video_dit_config, resolve=True)
        action_dit_config = OmegaConf.to_container(mc.action_dit_config, resolve=True)
        video_dit_config["use_gradient_checkpointing"] = False
        action_dit_config["use_gradient_checkpointing"] = False
        self.model = FastWAM.from_wan22_pretrained(
            device=self.device,
            torch_dtype=torch.bfloat16,
            model_id=str(mc.model_id),
            tokenizer_model_id=str(mc.tokenizer_model_id),
            tokenizer_max_len=int(mc.tokenizer_max_len),
            load_text_encoder=True,
            proprio_dim=None if mc.proprio_dim is None else int(mc.proprio_dim),
            speed_conditioned=self.speed_conditioned,
            redirect_common_files=bool(mc.redirect_common_files),
            video_dit_config=video_dit_config,
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=None,
            skip_dit_load_from_pretrain=True,
            mot_checkpoint_mixed_attn=False,
            video_train_shift=float(mc.video_scheduler.train_shift),
            video_infer_shift=float(mc.video_scheduler.infer_shift),
            video_num_train_timesteps=int(mc.video_scheduler.num_train_timesteps),
            action_train_shift=float(mc.action_scheduler.train_shift),
            action_infer_shift=float(mc.action_scheduler.infer_shift),
            action_num_train_timesteps=int(mc.action_scheduler.num_train_timesteps),
        )
        payload = torch.load(weights, map_location="cpu")
        self.model.mot.load_state_dict(payload["mot"], strict=True)
        if self.model.proprio_encoder is not None:
            self.model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        self.model.eval()

        self.processor = instantiate(dt.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(run_dir / "dataset_stats.json")))

        self.cuda_graph = cuda_graph
        self._infer_action: Callable[..., ActionResult] = self.model.infer_action
        self._request_context: Callable[[str], tuple[torch.Tensor, torch.Tensor]] = self._text_context
        if cuda_graph:
            from fastwam.models.wan22.graphed_infer import GraphedActionInfer

            shape = {
                "height": self.video_h,
                "width": self.video_w,
                "num_history": self.history_len,
                "history_width": history_frame_width(self.video_w, self.history_cameras),
                "context_len": self.context_len,
                "action_horizon": self.action_horizon,
                "num_inference_steps": self.num_inference_steps,
            }
            runner = GraphedActionInfer(self.model, **shape)
            self._infer_action = runner.infer_action
            self._request_context = functools.lru_cache(maxsize=32)(self._device_text_context)

    @torch.no_grad()
    def infer(self, obs: InferRequest) -> InferResponse:
        check_speed_request(self.speed_conditioned, obs)
        opst = np.ascontiguousarray(obs["observation.images.opst_cam"])
        wrist = np.ascontiguousarray(obs["observation.images.wrist_cam"])
        frames_np = np.concatenate([opst, wrist], axis=2)
        expected = (self.history_len + 1, self.video_h, self.video_w, 3)
        if frames_np.shape != expected:
            raise ValueError(f"expected images of shape {expected} after horizontal concat, got {frames_np.shape}")

        frames = torch.from_numpy(frames_np).to(self.device).permute(0, 3, 1, 2).float()
        frames = (frames * (2.0 / 255.0) - 1.0).to(self.model.torch_dtype)

        history_images = None
        history_is_pad = None
        if self.history_len > 0:
            history_images = select_history_cameras(frames[:-1], self.history_cameras)
            history_is_pad = torch.as_tensor(np.ascontiguousarray(obs["history_is_pad"], dtype=bool))

        state = np.ascontiguousarray(obs["observation.state"], dtype=np.float32).reshape(-1, 6)[-1]
        proprio = self._normalize_state(torch.from_numpy(state).reshape(1, -1))
        if self.speed_conditioned:
            speed = self._encode_speed(float(obs[SPEED_KEY]))
            proprio = torch.cat([proprio, torch.full_like(proprio[:, :1], speed)], dim=-1)

        context, context_mask = self._request_context(DEFAULT_PROMPT.format(task=obs["prompt"]))
        out = self._infer_action(
            prompt=None,
            context=context,
            context_mask=context_mask,
            input_image=frames[-1:],
            action_horizon=self.action_horizon,
            history_images=history_images,
            history_is_pad=history_is_pad,
            proprio=proprio,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
            rand_device="cpu",
        )
        chunk = self._denormalize_action(out["action"])
        chunk[:, 6] = np.clip(chunk[:, 6], 0.0, 1.0)
        return {"actions": chunk}

    def _text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = self.text_cache_dir / f"{digest}.t5_len{self.context_len}.{TEXT_CACHE_SUFFIX}.pt"
        if not path.is_file():
            logging.warning("prompt not in the training text cache, encoding it live: %r", prompt)
            return self.model.encode_prompt(prompt)
        payload = torch.load(path, map_location="cpu")
        context, mask = payload["context"], payload["mask"].bool()
        context[~mask] = 0.0
        return context, torch.ones_like(mask)

    def _device_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        context, mask = self._text_context(prompt)
        return context.to(self.device), mask.to(self.device)

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        key = self.processor.shape_meta["state"][0]["key"]
        batch = self.processor.action_state_transform({"state": {key: state}})
        return self.processor.normalizer.forward(batch)["state"][key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        key = self.processor.shape_meta["action"][0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][key]
        denorm = normalizer.backward(action[None].to(dtype=torch.float32, device="cpu"))
        return denorm[0].numpy().astype(np.float32)


def connect_metadata(policy: CheckpointPolicy) -> ConnectMetadata:
    return {
        "obs_delta_indices": policy.obs_delta_indices,
        "n_action_steps": policy.action_horizon,
        "euler_convention": policy.euler_convention,
        "bench_schema": policy.bench_schema,
        "speed_conditioned": policy.speed_conditioned,
        "fast_b": None,
    }


async def _handler(policy: CheckpointPolicy, frame_push_on: bool, websocket: _server.ServerConnection) -> None:
    logging.info("Connection from %s opened", websocket.remote_address)
    packer = Packer()
    await websocket.send(
        packer.pack(
            {
                **connect_metadata(policy),
                **frame_push.metadata(frame_push_on),
            }
        )
    )
    frames = frame_push.FrameStore(frame_push_on)
    while True:
        try:
            obs = frames.take(unpackb(await websocket.recv()))
            if obs is None:
                continue
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


async def _serve(policy: CheckpointPolicy, host: str, port: int, frame_push_on: bool) -> None:
    async with _server.serve(
        functools.partial(_handler, policy, frame_push_on), host, port, compression=None, max_size=None
    ) as server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="training output_dir (config.yaml + dataset_stats.json)")
    parser.add_argument(
        "--weights", default=None, help="weights .pt; default: the highest step in <run-dir>/checkpoints/weights"
    )
    parser.add_argument(
        "--fastwam-dir", required=True, help="FastWAM checkout (src/ is imported; checkpoints/ holds Wan2.2)"
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="training dataset root (meta/ is read); default: the run's dataset_dirs[0] under --fastwam-dir",
    )
    parser.add_argument(
        "--text-cache-dir",
        default=None,
        help="training T5 cache; default: the run's text_embedding_cache_dir under --fastwam-dir",
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="init-noise seed used on every replan")
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="replay infer_action from one CUDA graph (same action chunk bit for bit; FastWAM with graphed_infer)",
    )
    parser.add_argument(
        "--frame-push",
        action="store_true",
        help="accept frame pushes and frame refs (policy_servers/frame_push.py) and list frame_push in the connect "
        "metadata; off by default, which keeps the metadata and every request exactly as before",
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    run_dir = pathlib.Path(args.run_dir).expanduser().resolve()
    weights = pathlib.Path(args.weights).expanduser().resolve() if args.weights else latest_weights(run_dir)
    fastwam_dir = pathlib.Path(args.fastwam_dir).expanduser().resolve()
    dataset_root = pathlib.Path(args.dataset_root).expanduser().resolve() if args.dataset_root else None
    text_cache_dir = pathlib.Path(args.text_cache_dir).expanduser().resolve() if args.text_cache_dir else None
    logging.info("Loading %s (run %s, FastWAM at %s)", weights, run_dir, fastwam_dir)
    policy = CheckpointPolicy(
        run_dir,
        weights,
        fastwam_dir,
        dataset_root,
        text_cache_dir,
        args.num_inference_steps,
        args.seed,
        cuda_graph=args.cuda_graph,
    )
    logging.info(
        "Serving on 0.0.0.0:%d (obs_delta_indices=%s, n_action_steps=%d, num_inference_steps=%d, "
        "euler_convention=%s, bench_schema=%s, text_cache=%s (%d prompts), device=%s, cuda_graph=%s)",
        args.port,
        policy.obs_delta_indices,
        policy.action_horizon,
        policy.num_inference_steps,
        policy.euler_convention,
        policy.bench_schema,
        policy.text_cache_dir,
        len(list(policy.text_cache_dir.glob("*.pt"))),
        policy.device,
        policy.cuda_graph,
    )
    asyncio.run(_serve(policy, "0.0.0.0", args.port, args.frame_push))


if __name__ == "__main__":
    main()
