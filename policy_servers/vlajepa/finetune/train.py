"""Single-GPU VLA-JEPA fine-tune on a prepared bench dataset (see ``bench_data.py``)."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import re
import shutil
import sys
import time

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import history  # noqa: E402
import speed_conditioning  # noqa: E402
from bench_data import EXPECTED_SCHEMA, BenchDataset, Sample  # noqa: E402

CAMERA_STAMP = "bench_camera_vintage.json"
ACTION_STATS = "bench_action_stats.json"
REINIT_PREFIXES = ("action_model.action_encoder.", "action_model.action_decoder.")
STATE_INPUT_WEIGHT = "action_model.state_encoder.layer1.weight"


class EpochSampler(Sampler[int]):
    def __init__(self, size: int, seed: int, start: int) -> None:
        self.size, self.seed, self.start = size, seed, start

    def __iter__(self):
        position = self.start
        while True:
            epoch, offset = divmod(position, self.size)
            perm = np.random.default_rng((self.seed, epoch)).permutation(self.size)
            for index in perm[offset:]:
                yield int(index)
            position = (epoch + 1) * self.size

    def __len__(self) -> int:
        return 2**62


def collate(batch: list[Sample]) -> list[Sample]:
    return batch


def build_config(args: argparse.Namespace) -> DictConfig:
    cfg = OmegaConf.load(_HERE / "bench_vlajepa.yaml")
    cfg.framework.qwenvl.base_vlm = str(args.qwen)
    cfg.framework.vj2_model.base_encoder = str(args.vjepa)
    cfg.framework.qwenvl.attn_implementation = args.attn
    cfg.trainer.max_train_steps = args.steps
    cfg.trainer.num_warmup_steps = args.warmup
    cfg.trainer.restart = {"step": args.restart_step, "warmup": args.restart_warmup, "peak": args.restart_peak}
    cfg.trainer.micro_batch = args.micro_batch
    cfg.trainer.grad_accum = args.grad_accum
    cfg.trainer.seed = args.seed
    if args.speed_conditioned:
        cfg.framework.action_model.state_dim = speed_conditioning.STATE_DIM
    return cfg


_HOST_PATHS = (("framework", "qwenvl", "base_vlm"), ("framework", "vj2_model", "base_encoder"))


def _without_host_paths(cfg: DictConfig) -> dict:
    data = OmegaConf.to_container(cfg)
    assert isinstance(data, dict)
    for path in _HOST_PATHS:
        node = data
        for key in path[:-1]:
            node = node[key]
        node.pop(path[-1], None)
    return data


def _run_settings(cfg: DictConfig) -> dict:
    data = _without_host_paths(cfg)
    trainer = data["trainer"]
    trainer["global_batch"] = trainer.pop("micro_batch") * trainer.pop("grad_accum")
    return data


def restarted_schedule(base, restart_step: int, end: int, warmup: int, peak: float):
    floor = base(restart_step)
    steps = end - restart_step

    def factor(step: int) -> float:
        since = step - restart_step
        if since <= 0:
            return base(step)
        if since <= warmup:
            return floor + (peak - floor) * since / warmup
        if since <= steps:
            return peak + (floor - peak) * (since - warmup) / (steps - warmup)
        return floor

    return factor


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig) -> LambdaLR:
    from transformers import get_scheduler

    trainer, restart = cfg.trainer, cfg.trainer.restart
    cosine = get_scheduler(
        name="cosine_with_min_lr",
        optimizer=optimizer,
        num_warmup_steps=trainer.num_warmup_steps,
        num_training_steps=restart.step,
        scheduler_specific_kwargs=OmegaConf.to_container(trainer.scheduler_specific_kwargs),
    )
    factor = restarted_schedule(
        cosine.lr_lambdas[0], restart.step, trainer.max_train_steps, restart.warmup, restart.peak
    )
    return LambdaLR(optimizer, factor)


def check_restart(args: argparse.Namespace) -> None:
    if not 0 < args.restart_step <= args.steps:
        raise ValueError("need 0 < --restart-step <= --steps")
    if args.restart_step < args.steps and (
        not 0 < args.restart_warmup < args.steps - args.restart_step or not 0 < args.restart_peak <= 1
    ):
        raise ValueError("need 0 < --restart-warmup < --steps - --restart-step and 0 < --restart-peak <= 1")


def widen_state_input(weight: torch.Tensor, columns: int) -> torch.Tensor:
    widened = torch.zeros(weight.shape[0], columns, dtype=weight.dtype)
    widened[:, : weight.shape[1]] = weight
    return widened


def load_pretrained(model: torch.nn.Module, path: pathlib.Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    own = model.state_dict()
    kept, reinit = {}, []
    for key, value in checkpoint.items():
        if (
            key == STATE_INPUT_WEIGHT
            and key in own
            and own[key].shape[0] == value.shape[0]
            and own[key].shape[1] > value.shape[1]
        ):
            kept[key] = widen_state_input(value, own[key].shape[1])
            print(f"[bench] {key}: pretrained {tuple(value.shape)} widened to {tuple(own[key].shape)}", flush=True)
            continue
        if key in own and own[key].shape != value.shape:
            if not key.startswith(REINIT_PREFIXES):
                raise RuntimeError(f"{key}: checkpoint {tuple(value.shape)} vs model {tuple(own[key].shape)}")
            reinit.append(key)
            continue
        kept[key] = value
    missing, unexpected = model.load_state_dict(kept, strict=False)
    if unexpected or set(missing) != set(reinit):
        raise RuntimeError(f"pretrained load: missing {missing}, unexpected {unexpected}, re-initialised {reinit}")
    print(f"[bench] loaded {path}; re-initialised {reinit}", flush=True)


def checkpoint_steps(run_dir: pathlib.Path) -> list[int]:
    steps = []
    for p in (run_dir / "checkpoints").glob("step_*"):
        m = re.fullmatch(r"step_(\d+)", p.name)
        if m and (p / "train_state.pt").is_file():
            steps.append(int(m.group(1)))
    return sorted(steps)


def write_json(path: pathlib.Path, data: object) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def prepare_run_dir(run_dir: pathlib.Path, cfg: DictConfig, offsets: tuple[int, ...], data: BenchDataset) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    history.stamp_run_dir(run_dir, offsets, has_checkpoints=bool(checkpoint_steps(run_dir)))
    speed_conditioning.stamp_run_dir(
        run_dir, data.speed_norm is not None, has_checkpoints=bool(checkpoint_steps(run_dir))
    )
    stats = {"action_min": data.meta.action_min, "action_max": data.meta.action_max, "source": data.meta.source}
    stats_path = run_dir / ACTION_STATS
    if stats_path.is_file() and json.loads(stats_path.read_text()) != stats:
        raise RuntimeError(f"{stats_path} differs from the prepared dataset's statistics; use a fresh run dir")
    write_json(stats_path, stats)
    write_json(run_dir / CAMERA_STAMP, {"bench_schema": EXPECTED_SCHEMA})
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        saved_cfg = OmegaConf.load(config_path)
        saved = _run_settings(saved_cfg)
        if saved != _run_settings(cfg):
            raise RuntimeError(
                f"{config_path} differs from this invocation (other than weight paths and the micro-batch split); "
                f"resume with the run's own settings or use a fresh run dir.\nsaved: {saved}\nnow:   {_run_settings(cfg)}"
            )
        split = (saved_cfg.trainer.micro_batch, saved_cfg.trainer.grad_accum)
        if split != (cfg.trainer.micro_batch, cfg.trainer.grad_accum):
            print(
                f"[bench] micro-batch split {split[0]} x {split[1]} -> {cfg.trainer.micro_batch} x {cfg.trainer.grad_accum}"
            )
    OmegaConf.save(cfg, config_path)


def rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": [numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)), *numpy_state[2:]],
        "python": random.getstate(),
    }


def same_rng_state(a: dict, b: dict) -> bool:
    return (
        torch.equal(a["torch"], b["torch"])
        and len(a["cuda"]) == len(b["cuda"])
        and all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"], strict=True))
        and a["numpy"][0] == b["numpy"][0]
        and torch.equal(a["numpy"][1], b["numpy"][1])
        and a["numpy"][2:] == b["numpy"][2:]
        and a["python"] == b["python"]
    )


def init_wandb(args: argparse.Namespace, cfg: DictConfig, run_dir: pathlib.Path, offsets: tuple[int, ...]):
    import wandb

    id_file = run_dir / "wandb_run_id"
    if not id_file.is_file():
        id_file.write_text(f"{args.wandb_name}-{time.strftime('%Y%m%d-%H%M%S')}\n")
    before = rng_state()
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        id=id_file.read_text().strip(),
        resume="allow",
        dir=str(run_dir),
        tags=[t for t in args.wandb_tags.split(",") if t],
        config={
            "model": OmegaConf.to_container(cfg, resolve=True),
            "args": {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()},
            "exterior_history": list(offsets),
        },
    )
    if not same_rng_state(before, rng_state()):
        raise RuntimeError("wandb.init changed an RNG state; refusing to train with W&B on")
    return run


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    name, keys, *rest = state["numpy"]
    np.random.set_state((name, keys.cpu().numpy().astype(np.uint32), *rest))
    random.setstate(state["python"])


def optimizer_step(
    model, batches, grad_accum: int, optimizer, scheduler, clip: float, sums: dict[str, float]
) -> tuple[torch.Tensor, float]:
    data_time = 0.0
    for _ in range(grad_accum):
        t0 = time.time()
        batch = next(batches)
        data_time += time.time() - t0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model(batch)
            loss = sum(losses.values()) / grad_accum
        loss.backward()
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + v.item() / grad_accum
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm, data_time


def save(run_dir: pathlib.Path, step: int, model, optimizer, scheduler, keep: int) -> None:
    final = run_dir / "checkpoints" / f"step_{step}"
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    t0 = time.time()
    torch.save(model.state_dict(), tmp / "model.pt")
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step, "rng": rng_state()},
        tmp / "train_state.pt",
    )
    tmp.rename(final)
    print(f"[save] step {step} in {time.time() - t0:.1f}s -> {final}", flush=True)
    for old in checkpoint_steps(run_dir)[:-keep]:
        shutil.rmtree(run_dir / "checkpoints" / f"step_{old}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=pathlib.Path, help="output of bench_data.py")
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    parser.add_argument("--pretrained", required=True, type=pathlib.Path, help="VLA-JEPA-pretrain.pt")
    parser.add_argument("--qwen", required=True, type=pathlib.Path, help="Qwen3-VL-2B-Instruct dir")
    parser.add_argument("--vjepa", required=True, type=pathlib.Path, help="vjepa2-vitl-fpc64-256 dir")
    parser.add_argument("--steps", type=int, default=45000)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--restart-step", type=int, default=30000, help="the cosine ends and the LR restarts here")
    parser.add_argument("--restart-warmup", type=int, default=500, help="warm-up steps after the restart")
    parser.add_argument("--restart-peak", type=float, default=0.3, help="peak after the restart, x each LR")
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--keep", type=int, default=1, help="checkpoints kept (oldest deleted first)")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--max-steps-this-job", type=int, default=None, help="stop (after a save) once reached")
    parser.add_argument("--wandb-project", default=None, help="log the [train] fields to this W&B project")
    parser.add_argument("--wandb-name", default=None, help="W&B run name (default: the run dir name)")
    parser.add_argument("--wandb-tags", default="", help="comma-separated W&B tags")
    parser.add_argument("--speed-conditioned", action="store_true", help="append the episode's belt speed to the state")
    args = parser.parse_args()
    check_restart(args)

    offsets = history.offsets_from_env()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cfg = build_config(args)
    data = BenchDataset(args.data_root, offsets, speed_conditioned=args.speed_conditioned)
    run_dir = args.run_dir.resolve()
    prepare_run_dir(run_dir, cfg, offsets, data)
    print(
        f"[bench] history {offsets}, speed conditioning {data.speed_norm}, {len(data)} frames, {len(data.meta.episodes)} episodes",
        flush=True,
    )

    from starVLA.model.framework import build_framework
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    model = build_framework(cfg)
    resume = checkpoint_steps(run_dir)
    if not resume:
        load_pretrained(model, args.pretrained)
    model = model.cuda()
    for p in model.vj_encoder.parameters():
        p.requires_grad_(False)
    model.train()
    model.vj_encoder.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[bench] params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, trainable {trainable / 1e6:.1f}M")

    groups = [
        {**g, "params": [p for p in g["params"] if p.requires_grad]}
        for g in build_param_lr_groups(model=model, cfg=cfg)
    ]
    for g in groups:
        print(f"[bench] lr group {g['name']}: lr {g['lr']}, {sum(p.numel() for p in g['params']) / 1e6:.1f}M params")
    opt_cfg = cfg.trainer.optimizer
    optimizer = torch.optim.AdamW(
        groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(opt_cfg.betas),
        eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg)
    step = 0
    if resume:
        last = run_dir / "checkpoints" / f"step_{resume[-1]}"
        model.load_state_dict(torch.load(last / "model.pt", map_location="cuda", weights_only=True))
        state = torch.load(last / "train_state.pt", map_location="cuda", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
        print(f"[bench] resumed from {last} at step {step}", flush=True)

    per_step = args.micro_batch * args.grad_accum
    loader = DataLoader(
        data,
        batch_size=args.micro_batch,
        sampler=EpochSampler(len(data), args.seed, start=step * per_step),
        collate_fn=collate,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        multiprocessing_context="spawn" if args.workers > 0 else None,
    )
    batches = iter(loader)
    if resume:
        set_rng_state(state["rng"])
    print(
        f"[bench] global batch {per_step} = {args.micro_batch} x {args.grad_accum}; steps {step}..{args.steps}",
        flush=True,
    )
    if args.wandb_project:
        args.wandb_name = args.wandb_name or run_dir.name
        wandb_run = init_wandb(args, cfg, run_dir, offsets)
    else:
        wandb_run = None
    job_stop = args.steps if args.max_steps_this_job is None else min(args.steps, step + args.max_steps_this_job)
    t_log, data_time = time.time(), 0.0
    sums: dict[str, float] = {}
    while step < job_stop:
        grad_norm, waited = optimizer_step(
            model, batches, args.grad_accum, optimizer, scheduler, cfg.trainer.gradient_clipping, sums
        )
        data_time += waited
        step += 1
        if not all(math.isfinite(v) for v in sums.values()):
            raise RuntimeError(f"non-finite loss at step {step}: {sums}")
        if step % args.log_interval == 0 or step == job_stop:
            n = args.log_interval if step % args.log_interval == 0 else step % args.log_interval
            elapsed = time.time() - t_log
            if wandb_run is not None:
                wandb_run.log(
                    {
                        **{f"train/{k}": v / n for k, v in sums.items()},
                        "train/grad_norm": grad_norm.item(),
                        "train/lr_vlm": scheduler.get_last_lr()[0],
                        "perf/s_per_step": elapsed / n,
                        "perf/data_s_per_step": data_time / n,
                        "perf/gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    },
                    step=step,
                )
            print(
                f"[train] {time.time():.0f} step {step} "
                + " ".join(f"{k} {v / n:.4f}" for k, v in sums.items())
                + f" grad_norm {grad_norm.item():.3f} lr_vlm {scheduler.get_last_lr()[0]:.2e}"
                f" s/step {elapsed / n:.2f} data_s/step {data_time / n:.2f}"
                f" gpu_peak_gib {torch.cuda.max_memory_allocated() / 2**30:.1f}",
                flush=True,
            )
            t_log, data_time, sums = time.time(), 0.0, {}
        if step % args.save_interval == 0 or step == job_stop:
            save(run_dir, step, model, optimizer, scheduler, args.keep)
    if wandb_run is not None:
        wandb_run.finish()
    del batches, loader


if __name__ == "__main__":
    main()
