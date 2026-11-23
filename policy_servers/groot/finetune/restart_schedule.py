from __future__ import annotations

import dataclasses
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

SCHEDULE_STAMP = "bench_lr_schedule.json"


@dataclasses.dataclass(frozen=True)
class Restart:
    step: int
    warmup: int
    peak: float


def from_env() -> Restart | None:
    if not os.environ.get("BENCH_RESTART_STEP"):
        return None
    restart = Restart(
        step=int(os.environ["BENCH_RESTART_STEP"]),
        warmup=int(os.environ["BENCH_RESTART_WARMUP"]),
        peak=float(os.environ["BENCH_RESTART_PEAK"]),
    )
    if restart.step <= 0 or restart.warmup <= 0 or not 0 < restart.peak <= 1:
        raise SystemExit(
            f"need BENCH_RESTART_STEP > 0, BENCH_RESTART_WARMUP > 0 and 0 < BENCH_RESTART_PEAK <= 1, got {restart}"
        )
    return restart


def schedule_record(max_steps: int | None, warmup_ratio: float | None, restart: Restart | None) -> dict:
    return {
        "max_steps": max_steps,
        "warmup_ratio": warmup_ratio,
        "restart": None if restart is None else dataclasses.asdict(restart),
    }


def stamp_run_dir(run_dir: pathlib.Path, record: dict, existing_steps: tuple[int, ...]) -> None:
    path = run_dir / SCHEDULE_STAMP
    if path.is_file():
        recorded = json.loads(path.read_text())
        if recorded != json.loads(json.dumps(record)):
            raise SystemExit(f"{path}: the run trains on {recorded}; this launch asks for {record}")
        return
    if existing_steps:
        raise SystemExit(f"{run_dir} holds checkpoints {sorted(existing_steps)} but no {SCHEDULE_STAMP}")
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(path)


def schedule(base: Callable[[int], float], restart: Restart, end: int) -> Callable[[int], float]:
    floor = base(restart.step)
    steps = end - restart.step

    def factor(step: int) -> float:
        since = step - restart.step
        if since <= 0:
            return base(step)
        if since <= restart.warmup:
            return floor + (restart.peak - floor) * since / restart.warmup
        if since <= steps:
            return restart.peak + (floor - restart.peak) * (since - restart.warmup) / (steps - restart.warmup)
        return floor

    return factor


def install(restart: Restart) -> None:
    from torch.optim.lr_scheduler import LambdaLR
    from transformers import Trainer, get_scheduler

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if restart.step > num_training_steps or restart.step < num_training_steps <= restart.step + restart.warmup:
            raise SystemExit(
                f"--max-steps {num_training_steps} must be BENCH_RESTART_STEP or more than "
                f"BENCH_RESTART_STEP + BENCH_RESTART_WARMUP, got {restart}"
            )
        if self.lr_scheduler is None:
            optimizer = self.optimizer if optimizer is None else optimizer
            base = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.args.get_warmup_steps(restart.step),
                num_training_steps=restart.step,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self.lr_scheduler = LambdaLR(optimizer, schedule(base.lr_lambdas[0], restart, num_training_steps))
            self._created_lr_scheduler = True
        return self.lr_scheduler

    Trainer.create_scheduler = create_scheduler
