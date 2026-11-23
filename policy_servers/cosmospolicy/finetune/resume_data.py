from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from torch.utils.data import DistributedSampler

if TYPE_CHECKING:
    import pathlib

DATA_LAYOUT_FILE = "bench_data_layout.json"
SCHEDULE_FILE = "bench_lr_schedule.json"
SCHEDULER_FIELDS = ("cycle_lengths", "warm_up_steps", "f_start", "f_max", "f_min")
LAYOUT: dict[str, int | bool] = {}
RECORDED: dict[str, int | bool] = {}


class ResumableDistributedSampler(DistributedSampler):
    instances: list[ResumableDistributedSampler] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epoch_offset = 0
        self.skip = 0
        ResumableDistributedSampler.instances.append(self)

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch + self.epoch_offset)

    def __iter__(self):
        indices = list(super().__iter__())
        skip, self.skip = self.skip, 0
        if skip:
            print(f"[bench] data resumes in epoch {self.epoch} at sample {skip} of {len(indices)}", flush=True)
        return iter(indices[skip:])


def data_layout(config) -> dict[str, int | bool]:
    LAYOUT.clear()
    LAYOUT.update(
        {
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "batch_size": int(config.dataloader_train.batch_size),
            "grad_accum_iter": int(config.trainer.grad_accum_iter),
            "drop_last": bool(config.dataloader_train.drop_last),
        }
    )
    return dict(LAYOUT)


def _micro_batches_per_epoch(layout: dict[str, int | bool], per_rank: int) -> int:
    batch = int(layout["batch_size"])
    return per_rank // batch if layout["drop_last"] else -(-per_rank // batch)


def _global_position(
    layout: dict[str, int | bool], iteration: int, dataset_len: int, sampler_drop_last: bool
) -> tuple[int, int, int]:
    world, batch = int(layout["world_size"]), int(layout["batch_size"])
    per_rank = dataset_len // world if sampler_drop_last else -(-dataset_len // world)
    per_epoch = _micro_batches_per_epoch(layout, per_rank)
    epoch, micro = divmod(iteration * int(layout["grad_accum_iter"]), per_epoch)
    return epoch, min(micro * batch, per_rank) * world, min(per_epoch * batch, per_rank) * world


def restore_data_position(iteration: int) -> None:
    samplers = ResumableDistributedSampler.instances
    if len(samplers) != 1 or not LAYOUT:
        raise RuntimeError(f"refusing to resume at step {iteration}: cannot restore the data position")
    sampler, batch = samplers[0], int(LAYOUT["batch_size"])
    if RECORDED:
        args = (iteration, len(sampler.dataset), sampler.drop_last)
        here, recorded = _global_position(LAYOUT, *args), _global_position(RECORDED, *args)
        if here != recorded:
            raise RuntimeError(
                f"refusing to resume at step {iteration} under {LAYOUT}: (epoch, sample, epoch length) {here}"
                f" differs from {recorded} under the recorded {RECORDED}"
            )
        print(
            f"[bench] layout {LAYOUT} resumes the recorded {RECORDED} at the same samples: "
            f"epoch {here[0]}, sample {here[1]} of {here[2]}",
            flush=True,
        )
    epoch, micro = divmod(
        iteration * int(LAYOUT["grad_accum_iter"]), _micro_batches_per_epoch(LAYOUT, sampler.num_samples)
    )
    sampler.epoch_offset, sampler.skip = epoch, micro * batch


def _global_batch(layout: dict[str, int | bool]) -> int:
    return int(layout["world_size"]) * int(layout["batch_size"]) * int(layout["grad_accum_iter"])


def check_data_layout(run_dir: pathlib.Path, layout: dict[str, int | bool], resuming: bool) -> None:
    path = run_dir / DATA_LAYOUT_FILE
    if not resuming:
        path.write_text(json.dumps(layout, indent=1))
        return
    recorded = json.loads(path.read_text()) if path.is_file() else None
    RECORDED.clear()
    if recorded == layout:
        return
    if (
        recorded is None
        or _global_batch(recorded) != _global_batch(layout)
        or not (recorded["drop_last"] and layout["drop_last"])
    ):
        raise RuntimeError(f"refusing to resume {run_dir}: data layout {layout} differs from the recorded {recorded}")
    RECORDED.update(recorded)


def lr_schedule(config) -> dict:
    return {
        "max_iter": int(config.trainer.max_iter),
        **{field: list(config.scheduler[field]) for field in SCHEDULER_FIELDS},
    }


def check_lr_schedule(run_dir: pathlib.Path, schedule: dict, resuming: bool) -> None:
    path = run_dir / SCHEDULE_FILE
    if not resuming:
        path.write_text(json.dumps(schedule, indent=1))
        return
    recorded = json.loads(path.read_text()) if path.is_file() else None
    if recorded != json.loads(json.dumps(schedule)):
        raise RuntimeError(
            f"refusing to resume {run_dir}: learning-rate schedule {schedule} differs from the recorded {recorded}"
        )
