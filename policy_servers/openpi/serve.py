"""Bench-side openpi policy server."""

import openpi.training.checkpoints  # noqa: F401, I001

import argparse
import importlib.util
import logging
import pathlib
import socket
import sys
import time
import traceback

import websockets
import websockets.frames
from openpi_client import msgpack_numpy
from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

_BENCH_CONFIGS_PATH = pathlib.Path(__file__).resolve().parent / "finetune" / "bench_configs.py"
_spec = importlib.util.spec_from_file_location("openpi_bench_configs", _BENCH_CONFIGS_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"failed to locate bench_configs at {_BENCH_CONFIGS_PATH}")
_bench_configs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _bench_configs
_spec.loader.exec_module(_bench_configs)
_bench_configs.register()

HISTORY_KEYS = ("observation/exterior_image_1_left",)

_fp_spec = importlib.util.spec_from_file_location(
    "bench_frame_push", pathlib.Path(__file__).resolve().parent.parent / "frame_push.py"
)
assert _fp_spec is not None and _fp_spec.loader is not None
frame_push = importlib.util.module_from_spec(_fp_spec)
_fp_spec.loader.exec_module(frame_push)
FAST_GRAPH = "graph"


def _load_belt_speed_adapter():
    spec = importlib.util.spec_from_file_location(
        "bench_belt_speed_adapter", pathlib.Path(__file__).resolve().parent.parent / "belt_speed_adapter.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fast_pi05():
    spec = importlib.util.spec_from_file_location(
        "bench_fast_pi05", pathlib.Path(__file__).resolve().parent / "fast_pi05.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to locate fast_pi05.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_fh_spec = importlib.util.spec_from_file_location(
    "bench_fast_host", pathlib.Path(__file__).resolve().parent / "fast_host.py"
)
assert _fh_spec is not None and _fh_spec.loader is not None
fast_host = importlib.util.module_from_spec(_fh_spec)
_fh_spec.loader.exec_module(fast_host)

_bs_spec = importlib.util.spec_from_file_location(
    "bench_batched_siglip", pathlib.Path(__file__).resolve().parent / "batched_siglip.py"
)
assert _bs_spec is not None and _bs_spec.loader is not None
batched_siglip = importlib.util.module_from_spec(_bs_spec)
_bs_spec.loader.exec_module(batched_siglip)


class _FramePushServer(websocket_policy_server.WebsocketPolicyServer):
    def __init__(self, *args, frame_push: bool, belt_speed_adapter=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._frame_push = frame_push
        self._belt_speed_adapter = belt_speed_adapter

    async def _handler(self, websocket):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        frames = frame_push.FrameStore(self._frame_push)
        adapter = self._belt_speed_adapter.session() if self._belt_speed_adapter else None
        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                message = msgpack_numpy.unpackb(await websocket.recv())
                if adapter is not None:
                    adapter.observe(message)
                obs = frames.take(message)
                if obs is None:
                    continue
                if adapter is not None:
                    obs, adapter_info = adapter.fill(obs)

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time
                if adapter is not None:
                    action[self._belt_speed_adapter.reply_key] = adapter_info

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="openpi training config name (`bench`, or a stock config like pi05_libero)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="checkpoint dir — local path, or gs:// URL (auto-downloads). "
        "Can be a JAX orbax dir (contains params/) or PyTorch dir (contains model.safetensors).",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--pytorch-device",
        default=None,
        help="cpu / cuda / cuda:N — only used when the checkpoint is PyTorch. "
        "None = openpi default (cuda if available, else cpu).",
    )
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument(
        "--fast",
        choices=(FAST_GRAPH,),
        default=None,
        help="bench pi05 only: serve a converted PyTorch checkpoint (finetune/convert_torch.py) through fast_pi05.py, "
        "one CUDA graph per request. Default: upstream's policy.",
    )
    parser.add_argument(
        "--frame-push",
        action="store_true",
        help="accept frame pushes and frame refs (policy_servers/frame_push.py) and list frame_push in the connect "
        "metadata; off by default, which keeps the metadata and every request exactly as before",
    )
    parser.add_argument(
        "--belt-speed-adapter",
        default=None,
        help="a ground-truth-speed checkpoint served without the true speed: estimate belt_speed on this server "
        "from the pushed exterior frames with this adapter head file (policy_servers/belt_speed_adapter.py); "
        "needs --frame-push, and each request names its rollout with belt_speed_episode",
    )
    parser.add_argument(
        "--belt-speed-adapter-cuda-graph",
        action="store_true",
        help="--belt-speed-adapter: replay RAFT and the head from one CUDA graph",
    )
    args = parser.parse_args()
    if args.belt_speed_adapter and not args.frame_push:
        parser.error("--belt-speed-adapter reads the pushed frames: add --frame-push")
    if args.belt_speed_adapter_cuda_graph and not args.belt_speed_adapter:
        parser.error("--belt-speed-adapter-cuda-graph needs --belt-speed-adapter")
    if args.fast and args.config != _bench_configs.BENCH_CONFIG_NAME:
        parser.error(f"--fast serves the {_bench_configs.BENCH_CONFIG_NAME} config only")
    if args.fast and fast_host.enabled():
        parser.error(f"{fast_host.ENV}=1 speeds up the JAX policy; --fast={args.fast} serves the PyTorch one")
    if args.fast and batched_siglip.enabled():
        parser.error(f"{batched_siglip.ENV}=1 patches the JAX pi05; --fast={args.fast} serves the PyTorch one")

    logging.basicConfig(level=logging.INFO, force=True)

    served_schema = {}
    if args.config == _bench_configs.BENCH_CONFIG_NAME:
        served_schema = {
            _bench_configs.camera_vintage.SCHEMA_INFO_KEY: _bench_configs.check_checkpoint_camera_vintage(
                args.checkpoint_dir
            )
        }
        _bench_configs.check_checkpoint_history(args.checkpoint_dir)
        _bench_configs.check_checkpoint_gt_speed(args.checkpoint_dir)
        served_schema.update(_bench_configs.check_checkpoint_action_space(args.checkpoint_dir))

    train_config = _config.get_config(args.config)
    if train_config.model.model_type == _model.ModelType.PI0_FAST:
        decoder = _bench_configs.fast_decoder(train_config.model)
        logging.info("pi0-FAST action decoder: %s=%s", _bench_configs.FAST_DECODER_KEY, decoder)

    batched_siglip_on = batched_siglip.enabled()
    if batched_siglip_on:
        if train_config.model.model_type != _model.ModelType.PI05:
            raise ValueError(f"{batched_siglip.ENV}=1 is for pi05, not {train_config.model.model_type}")
        logging.info("%s=1: all images go through SigLIP in one call", batched_siglip.ENV)
    batched_siglip.install(batched_siglip_on)

    logging.info("Loading checkpoint from %s", args.checkpoint_dir)

    if args.fast == FAST_GRAPH:
        policy = _load_fast_pi05().create_fast_policy(
            train_config,
            args.checkpoint_dir,
            image_keys=_bench_configs.history_image_keys(_bench_configs.history.offsets_from_env()),
            default_prompt=args.default_prompt,
            device=args.pytorch_device or "cuda",
        )
        logging.info("--fast=%s: pi05 through fast_pi05.Pi05Graph", FAST_GRAPH)
    else:
        policy = _policy_config.create_trained_policy(
            train_config,
            args.checkpoint_dir,
            default_prompt=args.default_prompt,
            pytorch_device=args.pytorch_device,
        )
        if batched_siglip_on and policy._is_pytorch_model:
            raise ValueError(
                f"{batched_siglip.ENV}=1 patches the JAX pi05; {args.checkpoint_dir} is a PyTorch checkpoint"
            )

    fast_host_on = fast_host.enabled()
    if fast_host_on:
        logging.info("%s=1: batching, rng split and image conversion run inside the model's jit", fast_host.ENV)
        policy = fast_host.FastHostPolicy(policy)

    belt_speed_adapter = None
    if args.belt_speed_adapter:
        if not policy.metadata.get("speed_conditioned"):
            raise ValueError("--belt-speed-adapter is for a ground-truth-speed checkpoint (BENCH_GT_SPEED=1)")
        adapter_module = _load_belt_speed_adapter()
        belt_speed_adapter = adapter_module.BeltSpeedAdapter(
            args.belt_speed_adapter, HISTORY_KEYS[0], cuda_graph=args.belt_speed_adapter_cuda_graph
        )
        logging.info("belt-speed adapter %s: %s", args.belt_speed_adapter, belt_speed_adapter.info)

    hostname = socket.gethostname()
    logging.info("Serving on %s:%d", hostname, args.port)
    server = _FramePushServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        frame_push=args.frame_push,
        belt_speed_adapter=belt_speed_adapter,
        metadata={
            **policy.metadata,
            **served_schema,
            **frame_push.metadata(args.frame_push),
            **({fast_host.METADATA_KEY: True} if fast_host_on else {}),
            **({batched_siglip.METADATA_KEY: True} if batched_siglip_on else {}),
            **(belt_speed_adapter.metadata() if belt_speed_adapter else {}),
        },
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
