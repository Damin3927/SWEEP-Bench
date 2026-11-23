"""Upstream's ``cosmos_policy.scripts.train`` with the bench history stamp and base weights."""

from __future__ import annotations

import argparse
import os
import pathlib

import cosmos_policy.scripts.train as upstream_launch
import torch
from cosmos_policy._src.imaginaire.config import load_config
from cosmos_policy._src.imaginaire.utils import log
from cosmos_policy._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_policy._src.imaginaire.utils.device import Device
from cosmos_policy._src.predict2.checkpointer.dcp import DistributedCheckpointer
from cosmos_policy._src.predict2.utils.model_loader import load_model_state_dict_from_checkpoint
from cosmos_policy.scripts.train import launch
from huggingface_hub import hf_hub_download
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.tensor import DTensor

from .bench_config import BASE_FILE, BASE_REPO, BASE_REVISION, DEFAULT_BASE_CKPT, TRAIN_FRAMES
from .bench_dataset import bench_schema_name, read_info
from .history import offsets_from_env, stamp_run_dir
from .resume_data import (
    ResumableDistributedSampler,
    check_data_layout,
    check_lr_schedule,
    data_layout,
    lr_schedule,
    restore_data_position,
)

_upstream_load = DistributedCheckpointer.load


def _base_keys(model: torch.nn.Module) -> set[str]:
    return {
        k.replace("base_layer.", "").replace("base_model.model.", "")
        for k in model.state_dict()
        if k.startswith("net.") and "lora_" not in k
    }


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _load_base_into_sharded(model: torch.nn.Module, base: dict[str, torch.Tensor]) -> None:
    if getattr(model.config, "use_lora", False):
        raise NotImplementedError("base .pt into an FSDP-sharded LoRA model (key mapping not implemented)")
    net_state = {k.removeprefix("net."): v for k, v in base.items() if k.startswith("net.")}
    result = set_model_state_dict(model.net, net_state, options=StateDictOptions(full_state_dict=True, strict=False))
    missing = [k for k in result.missing_keys if not k.endswith("_extra_state")]
    if missing:
        raise RuntimeError(f"base .pt left {len(missing)} sharded weights unset, e.g. {missing[:5]}")


def _load_with_base_pt(self, model, optimizer=None, scheduler=None, grad_scaler=None) -> int:
    resuming = self._read_latest_checkpoint_file() is not None
    iteration = _upstream_load(self, model, optimizer, scheduler, grad_scaler)
    if resuming:
        restore_data_position(iteration)
    if resuming or not self.load_path or not str(self.load_path).endswith(".pt"):
        return iteration
    base = torch.load(self.load_path, map_location="cpu", weights_only=False)
    missing = sorted(_base_keys(model) - set(base))
    if missing:
        raise RuntimeError(f"{self.load_path} lacks {len(missing)} model weights, e.g. {missing[:5]}")
    probe = next(k for k in base if k.endswith(".weight"))
    before = _full(model.state_dict()[_model_key(model, probe)]).detach().float().cpu().clone()
    if any(isinstance(q, DTensor) for q in model.net.parameters()):
        _load_base_into_sharded(model, base)
    else:
        load_model_state_dict_from_checkpoint(model, None, str(self.load_path))
    after = _full(model.state_dict()[_model_key(model, probe)]).detach().float().cpu()
    if not torch.equal(after, base[probe].float()) or torch.equal(before, after):
        raise RuntimeError(f"loading {self.load_path} did not set {probe}")
    log.critical(f"[bench] loaded base weights from {self.load_path} ({len(base)} tensors)")
    return iteration


def _cpu_affinity_within_job(self: Device) -> list[int]:
    allowed = os.sched_getaffinity(0)
    return sorted(set(_upstream_affinity(self)) & allowed) or sorted(allowed)


_upstream_affinity = Device.get_cpu_affinity


def _model_key(model: torch.nn.Module, base_key: str) -> str:
    keys = model.state_dict().keys()
    if base_key in keys:
        return base_key
    lora = base_key.replace(".weight", ".base_layer.weight")
    if lora in keys:
        return lora
    raise KeyError(base_key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = load_config(args.config, args.opts, enable_one_logger=False)
    if config.checkpoint.load_path == DEFAULT_BASE_CKPT:
        config.checkpoint.load_path = hf_hub_download(BASE_REPO, BASE_FILE, revision=BASE_REVISION)
    elif config.checkpoint.load_path:
        config.checkpoint.load_path = get_checkpoint_path(config.checkpoint.load_path)
    if int(os.environ.get("RANK", "0")) == 0:
        dataset_dir = pathlib.Path(os.environ["BENCH_DATASET_DIR"])
        frames = read_info(dataset_dir)["total_frames"]
        if frames != TRAIN_FRAMES:
            raise ValueError(f"{dataset_dir} has {frames} frames; the recipe is set for {TRAIN_FRAMES}")
        stamp_run_dir(pathlib.Path(config.job.path_local), offsets_from_env(), bench_schema_name(dataset_dir))
    run_dir = pathlib.Path(config.job.path_local)
    resuming = (run_dir / "checkpoints" / "latest_checkpoint.txt").is_file()
    layout = data_layout(config)
    if int(os.environ.get("RANK", "0")) == 0:
        check_data_layout(run_dir, layout, resuming)
        check_lr_schedule(run_dir, lr_schedule(config), resuming)
    DistributedCheckpointer.load = _load_with_base_pt
    Device.get_cpu_affinity = _cpu_affinity_within_job
    upstream_launch.DistributedSampler = ResumableDistributedSampler
    launch(config, argparse.Namespace(config=args.config, opts=args.opts, dryrun=False))


if __name__ == "__main__":
    main()
