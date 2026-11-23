from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import LambdaLR, LRScheduler, MultiStepLR

if TYPE_CHECKING:
    import pathlib

    import torch

DECAY = 0.1


@dataclasses.dataclass(frozen=True)
class Anneal:
    start: int
    steps: int
    end: float


def anneal_to_end(start: int | None, end: float | None, decay_step: int, max_steps: int) -> Anneal | None:
    if (start is None) != (end is None):
        raise ValueError("--anneal-from and --anneal-end go together")
    if start is None or end is None:
        return None
    if not decay_step <= start < max_steps or end <= 0:
        raise ValueError("need --decay-step <= --anneal-from < --max-steps and a positive --anneal-end")
    return Anneal(start=start, steps=max_steps - start, end=end)


def lr_factor(step: int, decay_step: int, anneal: Anneal | None) -> float:
    if step < decay_step:
        return 1.0
    if anneal is None or step < anneal.start:
        return DECAY
    done = min(step - anneal.start, anneal.steps) / anneal.steps
    return DECAY + (anneal.end - DECAY) * done


def build_scheduler(optimizer: torch.optim.Optimizer, decay_step: int, anneal: Anneal | None) -> LRScheduler:
    if anneal is None:
        return MultiStepLR(optimizer, milestones=[decay_step], gamma=DECAY)
    return LambdaLR(optimizer, lambda step: lr_factor(step, decay_step, anneal))


LAYOUT_FILE = "bench_layout_changes.jsonl"
LAYOUT_KEYS = ("batch_size", "grad_accum")


def resume_layout_change(recorded: dict, current: dict) -> dict[str, list] | None:
    diff = {k: [recorded.get(k), v] for k, v in current.items() if recorded.get(k) != v}
    layout = {k: diff.pop(k) for k in LAYOUT_KEYS if k in diff}
    if diff:
        raise RuntimeError(f"the run was trained with a different config (recorded, now): {diff}")
    if not layout:
        return None
    before = recorded["batch_size"] * recorded["grad_accum"]
    after = current["batch_size"] * current["grad_accum"]
    if before != after:
        raise RuntimeError(f"a resume may change the micro-batch layout, not the global batch ({before} -> {after})")
    return layout


def log_layout(run_dir: pathlib.Path, stamped: dict, current: dict, world: int, from_step: int) -> dict | None:
    now = {
        "world": world,
        "micro_batches_per_rank": current["grad_accum"] // world,
        **{k: current[k] for k in LAYOUT_KEYS},
    }
    path = run_dir / LAYOUT_FILE
    lines = path.read_text().splitlines() if path.is_file() else []
    if lines:
        if json.loads(lines[-1])["now"] == now:
            return None
    elif all(stamped[k] == current[k] for k in LAYOUT_KEYS):
        return None
    record = {
        "from_step": from_step,
        "now": now,
        "stamped": {k: stamped[k] for k in LAYOUT_KEYS},
        "global_batch": current["batch_size"] * current["grad_accum"],
    }
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record
