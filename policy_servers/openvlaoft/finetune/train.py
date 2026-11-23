"""OpenVLA-OFT LoRA fine-tune on a bench LeRobot dataset, resumable."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import pathlib
import re
import shutil
import sys
import time
from typing import TYPE_CHECKING, TypedDict, cast

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bench_common as bc  # noqa: E402  (must precede every prismatic import)
import bench_data  # noqa: E402
import bench_schedule as schedule  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict  # noqa: E402
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor  # noqa: E402
from prismatic.models.action_heads import L1RegressionActionHead  # noqa: E402
from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # noqa: E402
from prismatic.models.projectors import ProprioProjector  # noqa: E402
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask  # noqa: E402
from prismatic.util.data_utils import PaddedCollatorForActionPrediction  # noqa: E402
from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: E402
from prismatic.vla.datasets.datasets import RLDSBatchTransform  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

    from PIL import Image

CHECKPOINTS = "checkpoints"
STEP_DIR = re.compile(r"step_(\d{6})")


@dataclasses.dataclass(frozen=True)
class RunConfig:
    dataset: str
    max_episodes: int | None
    base_model: str
    exterior_history: tuple[int, ...]
    num_images: int
    num_actions_chunk: int
    action_dim: int
    proprio_dim: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    decay_step: int
    anneal: schedule.Anneal | None
    max_steps: int
    lora_rank: int
    image_aug: bool
    seed: int


class Batch(TypedDict):
    pixel_values: torch.Tensor
    proprio: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    actions: torch.Tensor


class Collator:
    def __init__(self, model_max_length: int, pad_token_id: int) -> None:
        self._collate = PaddedCollatorForActionPrediction(model_max_length, pad_token_id, padding_side="right")

    def __call__(self, samples: list[bench_data.Sample]) -> Batch:
        out = self._collate(cast("list[dict[str, torch.Tensor]]", samples))
        fields = ("pixel_values", "proprio", "input_ids", "attention_mask", "labels", "actions")
        for key in fields:
            if not isinstance(out[key], torch.Tensor):
                raise TypeError(f"collator returned {key} as {type(out[key]).__name__}, expected a tensor")
        return {
            "pixel_values": out["pixel_values"],
            "proprio": out["proprio"],
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
            "labels": out["labels"],
            "actions": out["actions"],
        }


def image_processor(processor: PrismaticProcessor) -> PrismaticImageProcessor:
    image = processor.__dict__.get("image_processor")
    if not isinstance(image, PrismaticImageProcessor):
        raise TypeError(f"expected a PrismaticImageProcessor, got {type(image).__name__}")
    return image


class ApplyTransform:
    def __init__(self, processor: PrismaticProcessor) -> None:
        self._image = image_processor(processor)

    def __call__(self, img: Image.Image, **kwargs: str) -> torch.Tensor:
        return self._image.apply_transform(img)


def register_auto_classes() -> None:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def build_model(base: str, num_images: int, lora_rank: int, device: torch.device):
    vla = OpenVLAForActionPrediction.from_pretrained(base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    vla = vla.to(device)
    vla.vision_backbone.set_num_images_in_input(num_images)
    lora = LoraConfig(
        r=lora_rank,
        lora_alpha=min(lora_rank, 16),
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora)
    proprio_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=bc.PROPRIO_DIM).to(device)
    action_head = L1RegressionActionHead(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=bc.ACTION_DIM)
    action_head = action_head.to(torch.bfloat16).to(device)
    return vla, proprio_projector, action_head


def num_patches(vla, use_proprio: bool = True) -> int:
    return vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input() + int(use_proprio)


def l1_loss(
    vla, action_head, proprio_projector, batch: Batch, device: torch.device
) -> tuple[torch.Tensor, dict[str, float]]:
    gt = batch["actions"].to(device).to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
            labels=batch["labels"].to(device),
            output_hidden_states=True,
            proprio=batch["proprio"].to(device),
            proprio_projector=proprio_projector,
            use_film=False,
        )
    token_ids = batch["labels"][:, 1:].to(device)
    current_mask = get_current_action_mask(token_ids)
    next_mask = get_next_actions_mask(token_ids)
    text_hidden = output.hidden_states[-1][:, num_patches(vla) : -1]
    b = batch["input_ids"].shape[0]
    hidden = text_hidden[current_mask | next_mask].reshape(b, bc.NUM_ACTIONS_CHUNK * bc.ACTION_DIM, -1)
    pred = action_head.predict_action(hidden.to(torch.bfloat16))
    loss = torch.nn.L1Loss()(gt, pred)
    with torch.no_grad():
        metrics = {
            "loss": loss.item(),
            "curr_action_l1": torch.nn.L1Loss()(gt[:, 0], pred[:, 0]).item(),
            "next_actions_l1": torch.nn.L1Loss()(gt[:, 1:], pred[:, 1:]).item(),
        }
    return loss, metrics


def step_dirs(run_dir: pathlib.Path) -> dict[int, pathlib.Path]:
    root = run_dir / CHECKPOINTS
    if not root.is_dir():
        return {}
    return {int(m.group(1)): p for p in root.iterdir() if (m := STEP_DIR.fullmatch(p.name)) and p.is_dir()}


def save(run_dir, step, vla, proprio_projector, action_head, optimizer, scheduler, keep_interval) -> None:
    final = run_dir / CHECKPOINTS / f"step_{step:06d}"
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    t0 = time.time()
    vla.save_pretrained(tmp / "lora_adapter")
    torch.save(proprio_projector.state_dict(), tmp / "proprio_projector.pt")
    torch.save(action_head.state_dict(), tmp / "action_head.pt")
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        tmp / "train_state.pt",
    )
    shutil.copy2(run_dir / bc.STATS_FILE, tmp / bc.STATS_FILE)
    tmp.rename(final)
    for old, path in step_dirs(run_dir).items():
        if old < step and (keep_interval <= 0 or old % keep_interval != 0):
            shutil.rmtree(path)
    logging.info("[save] step %d -> %s in %.1fs", step, final, time.time() - t0)


def load(path, vla, proprio_projector, action_head, optimizer, scheduler) -> int:
    set_peft_model_state_dict(vla, load_file(path / "lora_adapter" / "adapter_model.safetensors"))
    proprio_projector.load_state_dict(torch.load(path / "proprio_projector.pt", map_location="cpu"))
    action_head.load_state_dict(torch.load(path / "action_head.pt", map_location="cpu"))
    state = torch.load(path / "train_state.pt", map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["step"])


def stamp_config(run_dir: pathlib.Path, cfg: RunConfig, existing: dict[int, pathlib.Path]) -> dict:
    path = run_dir / bc.RUN_CONFIG_FILE
    current = json.loads(json.dumps(dataclasses.asdict(cfg)))
    if path.is_file():
        recorded = json.loads(path.read_text())
        try:
            schedule.resume_layout_change(recorded, current)
        except RuntimeError as e:
            raise RuntimeError(f"{run_dir}: {e}") from None
        return recorded
    if existing:
        raise RuntimeError(f"{run_dir} holds checkpoints {sorted(existing)} but no {bc.RUN_CONFIG_FILE}")
    path.write_text(json.dumps(current, indent=2))
    return current


def init_distributed() -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1, torch.device("cuda")
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    if dist.get_backend() != "nccl" or dist.get_world_size() != world:
        raise RuntimeError(f"process group is {dist.get_backend()} x {dist.get_world_size()}, expected nccl x {world}")
    return dist.get_rank(), world, torch.device("cuda", local)


def coalesced(tensors: list[torch.Tensor], op: Callable[[torch.Tensor], object]) -> None:
    for dtype in sorted({t.dtype for t in tensors}, key=str):
        group = [t for t in tensors if t.dtype == dtype]
        flat = torch.cat([t.reshape(-1).float() for t in group])
        op(flat)
        offset = 0
        for t in group:
            t.copy_(flat[offset : offset + t.numel()].view_as(t))
            offset += t.numel()


def check_ranks_agree(params: list[torch.Tensor], world: int, step: int) -> None:
    total = torch.stack([q.detach().double().sum() for q in params]).sum()
    gathered = [torch.empty_like(total) for _ in range(world)]
    dist.all_gather(gathered, total)
    if any(not torch.equal(g, gathered[0]) for g in gathered):
        raise RuntimeError(f"step {step}: trainable weights differ across ranks: {[g.item() for g in gathered]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--stats", type=pathlib.Path, required=True, help="dataset_statistics.json from prep.py")
    p.add_argument("--base", required=True, help="openvla/openvla-7b snapshot dir")
    p.add_argument("--run-dir", type=pathlib.Path, required=True)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8, help="per micro-batch")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--decay-step", type=int, default=100_000)
    p.add_argument("--max-steps", type=int, default=150_000)
    p.add_argument("--save-interval", type=int, default=2_000)
    p.add_argument("--keep-interval", type=int, default=10_000, help="saves at multiples of this are kept")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--no-image-aug", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--wandb-project", default=None, help="log to W&B (WANDB_MODE decides online/offline)")
    p.add_argument("--anneal-from", type=int, default=None, help="the step the linear anneal from 0.1x starts at")
    p.add_argument("--anneal-end", type=float, default=None, help="factor of the base rate at --max-steps, e.g. 0.01")
    args = p.parse_args()
    try:
        anneal = schedule.anneal_to_end(args.anneal_from, args.anneal_end, args.decay_step, args.max_steps)
    except ValueError as e:
        p.error(str(e))

    rank, world, device = init_distributed()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING, format="%(asctime)s %(message)s", force=True
    )
    if args.grad_accum % world:
        raise ValueError(f"--grad-accum {args.grad_accum} must be a multiple of the {world} processes")
    offsets = bc.history.offsets_from_env()
    cfg = RunConfig(
        dataset=args.dataset_root.resolve().name,
        max_episodes=args.max_episodes,
        base_model=pathlib.Path(args.base).name,
        exterior_history=offsets,
        num_images=bc.num_images(offsets),
        num_actions_chunk=bc.NUM_ACTIONS_CHUNK,
        action_dim=bc.ACTION_DIM,
        proprio_dim=bc.PROPRIO_DIM,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        decay_step=args.decay_step,
        anneal=anneal,
        max_steps=args.max_steps,
        lora_rank=args.lora_rank,
        image_aug=not args.no_image_aug,
        seed=args.seed,
    )
    run_dir = args.run_dir.resolve()
    stats = bc.read_stats(args.stats)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        existing = step_dirs(run_dir)
        bc.history.stamp_run_dir(run_dir, offsets, existing_steps=tuple(sorted(existing)))
        stamped = stamp_config(run_dir, cfg, existing)
        record = schedule.log_layout(run_dir, stamped, dataclasses.asdict(cfg), world, max(existing) if existing else 0)
        if record:
            logging.info("[bench] resume with a new micro-batch layout: %s", record)
        if (run_dir / bc.STATS_FILE).is_file():
            if bc.read_stats(run_dir / bc.STATS_FILE) != stats:
                raise RuntimeError(f"{args.stats} differs from the {bc.STATS_FILE} this run trained with")
        else:
            bc.write_stats(run_dir, stats)
        logging.info("run %s: %s", run_dir, cfg)
    if world > 1:
        dist.barrier()
    existing = step_dirs(run_dir)

    torch.manual_seed(args.seed)
    register_auto_classes()
    processor = PrismaticProcessor.from_pretrained(args.base)
    vla, proprio_projector, action_head = build_model(args.base, cfg.num_images, args.lora_rank, device)
    vla.print_trainable_parameters()
    trainable = [q for m in (vla, action_head, proprio_projector) for q in m.parameters() if q.requires_grad]
    logging.info("trainable params: %d", sum(q.numel() for q in trainable))
    if world > 1:
        coalesced([q.data for q in trainable], lambda t: dist.broadcast(t, 0))
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    scheduler = schedule.build_scheduler(optimizer, args.decay_step, anneal)
    step = 0
    if existing:
        latest = max(existing)
        step = load(existing[latest], vla, proprio_projector, action_head, optimizer, scheduler)
        logging.info("resumed from %s at step %d", existing[latest], step)
    if step >= args.max_steps:
        logging.info("already at step %d >= --max-steps %d", step, args.max_steps)
        return

    frames = bench_data.BenchFrames(args.dataset_root, max_episodes=args.max_episodes)
    batch_transform = RLDSBatchTransform(
        ActionTokenizer(processor.tokenizer),
        processor.tokenizer,
        image_transform=ApplyTransform(processor),
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
    )
    dataset = bench_data.BenchSampleDataset(
        frames=frames,
        stats=stats,
        batch_transform=batch_transform,
        image_transform=ApplyTransform(processor),
        exterior_history=offsets,
        seed=args.seed,
        image_aug=cfg.image_aug,
    )
    samples_per_step = args.batch_size * args.grad_accum
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=bench_data.StreamSampler(
            step * samples_per_step, args.max_steps * samples_per_step, args.batch_size, rank, world
        ),
        collate_fn=Collator(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id),
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    logging.info(
        "%d frames / %d episodes, %d samples per step (%.3f epochs to step %d)",
        len(frames),
        frames.num_episodes,
        samples_per_step,
        args.max_steps * samples_per_step / len(frames),
        args.max_steps,
    )
    local_accum = args.grad_accum // world
    logging.info("%d processes x %d micro-batches of %d per step", world, local_accum, args.batch_size)

    wandb = None
    if args.wandb_project and rank == 0:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=run_dir.name,
            id=run_dir.name,
            resume="allow",
            config=dataclasses.asdict(cfg),
        )

    vla.train()
    optimizer.zero_grad()
    window: list[dict[str, float]] = []
    t_log = time.time()
    for micro, batch in enumerate(loader):
        loss, metrics = l1_loss(vla, action_head, proprio_projector, batch, device)
        (loss / args.grad_accum).backward()
        window.append(metrics)
        if (micro + 1) % local_accum:
            continue
        if world > 1:
            coalesced([q.grad for q in trainable if q.grad is not None], dist.all_reduce)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1
        if step % args.log_interval == 0 or step == args.max_steps:
            dt = (time.time() - t_log) / args.log_interval
            mean = {k: float(np.mean([m[k] for m in window])) for k in window[0]}
            if world > 1:
                check_ranks_agree(trainable, world, step)
                summed = torch.tensor(list(mean.values()), device=device, dtype=torch.float64)
                dist.all_reduce(summed)
                mean = {k: v / world for k, v in zip(mean, summed.tolist(), strict=True)}
            logging.info(
                "step %d loss %.4f curr_l1 %.4f next_l1 %.4f lr %.2e | %.2f s/step %.1f samples/s | "
                "gpu alloc %.1f GiB peak %.1f GiB",
                step,
                mean["loss"],
                mean["curr_action_l1"],
                mean["next_actions_l1"],
                scheduler.get_last_lr()[0],
                dt,
                samples_per_step / dt,
                torch.cuda.memory_allocated() / 2**30,
                torch.cuda.max_memory_allocated() / 2**30,
            )
            if wandb is not None:
                wandb.log({**mean, "lr": scheduler.get_last_lr()[0], "s_per_step": dt}, step=step)
            window.clear()
            t_log = time.time()
        if step % args.save_interval == 0 or step == args.max_steps:
            if rank == 0:
                save(run_dir, step, vla, proprio_projector, action_head, optimizer, scheduler, args.keep_interval)
            if world > 1:
                dist.barrier()
        if step >= args.max_steps:
            break
    logging.info("done at step %d", step)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
