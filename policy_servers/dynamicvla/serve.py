"""Bench-side DynamicVLA policy server."""

import argparse
import asyncio
import functools
import importlib.util
import json
import logging
import pathlib
import sys
import traceback

import msgpack
import numpy as np
import torch
import torchvision.transforms.v2.functional as F
import websockets.asyncio.server as _server
import websockets.frames

_UPSTREAM_DIR = pathlib.Path(__file__).resolve().parent / "upstream"
sys.path.insert(0, str(_UPSTREAM_DIR))

import utils.helpers  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "finetune"))
import speed_token  # noqa: E402


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


@functools.cache
def _cuda_graph_module():
    spec = importlib.util.spec_from_file_location(
        "dvla_cuda_graph", pathlib.Path(__file__).resolve().parent / "cuda_graph.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EULER_STAMP = "bench_euler_convention.json"
EULER_RX_WRAPPED = "rx_wrapped"
EULER_XZ_WRAPPED = "xz_wrapped"


def _euler_convention(ckpt_dir: pathlib.Path) -> str:
    stamp = next(
        (p / EULER_STAMP for p in (ckpt_dir, *ckpt_dir.parents) if (p / EULER_STAMP).is_file()),
        None,
    )
    if stamp is None:
        logging.warning(
            "no %s near %s: assuming upstream's DOM convention %s (rX and rZ wrapped); "
            "finetune/train.py stamps its runs.",
            EULER_STAMP,
            ckpt_dir,
            EULER_XZ_WRAPPED,
        )
        return EULER_XZ_WRAPPED
    convention = str(json.loads(stamp.read_text())["euler_convention"])
    logging.info("euler convention %s (from %s)", convention, stamp)
    return convention


class CheckpointPolicy:
    def __init__(
        self,
        ckpt_dir: pathlib.Path,
        cuda_graph: bool = False,
        warm_prompts: tuple[str, ...] = (),
    ) -> None:
        cfg_file = ckpt_dir / "config.json"
        with open(cfg_file) as fp:
            model_cfg = json.load(fp)
        vla_cfg = utils.helpers.get_policy_cfg(cfg_file=cfg_file)
        policy_class = utils.helpers.get_policy_class(model_cfg["type"])
        stamp = speed_token.find_stamp(ckpt_dir)
        self.speed_conditioned = stamp is not None
        if stamp is not None:
            speed_token.read_stamp(stamp)
            speed_token.patch_prepare_state(policy_class)
            logging.info("speed-conditioned checkpoint (from %s)", stamp)
        self.policy = policy_class.from_pretrained(ckpt_dir, config=vla_cfg)
        self.policy.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy.to(self.device)
        self.policy.reset()

        dts = (model_cfg.get("delta_timestamps") or {}).get("observation") or [0]
        if len(dts) != self.policy.config.n_obs_steps:
            raise ValueError(
                f"checkpoint has n_obs_steps={self.policy.config.n_obs_steps} but "
                f"delta_timestamps {dts}; refusing to serve an inconsistent config"
            )
        self.obs_delta_indices = [int(dt) for dt in dts]
        self.euler_convention = _euler_convention(ckpt_dir)
        self.graph = None
        if cuda_graph:
            if self.speed_conditioned or self.device != "cuda":
                raise ValueError("--cuda-graph needs a CUDA device and a checkpoint that is not speed-conditioned")
            self.graph = _cuda_graph_module().GraphPolicy(self.policy, self.device, warm_prompts)

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        cfg = self.policy.config
        speed_token.check_request(self.speed_conditioned, obs)
        if self.graph is not None:
            return {"actions": self.graph.infer(obs)}
        batch = {"task": [obs["prompt"]]}
        if self.speed_conditioned:
            batch[speed_token.SPEED_KEY] = torch.tensor([float(obs[speed_token.SPEED_KEY])], device=self.device)
        for key, feat in cfg.image_features.items():
            imgs = torch.from_numpy(np.ascontiguousarray(obs[key]))
            imgs = imgs.to(self.device).permute(0, 3, 1, 2).float() / 255.0
            imgs = F.resize(imgs, list(feat.shape[-2:]))
            batch[key] = imgs.contiguous()[None]
        state = torch.from_numpy(np.ascontiguousarray(obs["observation.state"], dtype=np.float32))
        state = state[None].to(self.device)
        batch["observation.state"] = state

        actions = self.policy.predict_action_chunk(batch)
        if cfg.use_delta_action:
            action_dim = actions.shape[-1] - 1
            actions[..., :action_dim] += state[:, -1:, :action_dim]
        return {"actions": actions[0].float().cpu().numpy()}

    def metadata(self) -> dict:
        return {
            "obs_delta_indices": self.obs_delta_indices,
            "n_action_steps": self.policy.config.n_action_steps,
            "euler_convention": self.euler_convention,
            "speed_conditioned": self.speed_conditioned,
            "cuda_graph": self.graph is not None,
            "fast_b": None,
        }


async def _handler(policy: CheckpointPolicy, frame_push_on: bool, websocket: _server.ServerConnection) -> None:
    logging.info("Connection from %s opened", websocket.remote_address)
    packer = Packer()
    await websocket.send(packer.pack({**policy.metadata(), **frame_push.metadata(frame_push_on)}))
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
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="fine-tuned checkpoint dir (config.json + model.safetensors), e.g. upstream/runs/checkpoints/<exp>",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="replay each request from a CUDA graph (cuda_graph.py); actions are bitwise equal to the default path",
    )
    parser.add_argument(
        "--warm-prompts",
        nargs="+",
        default=[],
        help="with --cuda-graph: prompts to capture graphs for at startup (other prompt lengths capture on first use)",
    )
    parser.add_argument(
        "--frame-push",
        action="store_true",
        help="accept frame pushes and frame refs (policy_servers/frame_push.py) and list frame_push in the connect "
        "metadata; off by default, which keeps the metadata and every request exactly as before",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    ckpt_dir = pathlib.Path(args.checkpoint_dir).expanduser().resolve()
    logging.info("Loading checkpoint from %s", ckpt_dir)
    policy = CheckpointPolicy(
        ckpt_dir,
        cuda_graph=args.cuda_graph,
        warm_prompts=tuple(args.warm_prompts),
    )
    logging.info(
        "Serving on 0.0.0.0:%d (obs_delta_indices=%s, n_action_steps=%d, euler_convention=%s, "
        "speed_conditioned=%s, device=%s, cuda_graph=%s)",
        args.port,
        policy.obs_delta_indices,
        policy.policy.config.n_action_steps,
        policy.euler_convention,
        policy.speed_conditioned,
        policy.device,
        policy.graph is not None,
    )
    asyncio.run(_serve(policy, "0.0.0.0", args.port, args.frame_push))


if __name__ == "__main__":
    main()
