from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence, Sized
    from typing import NotRequired

STATE_FILE = "state.pt"
_DATA_ENV = ("BENCH_TEST_FRACTION", "BENCH_SPLIT_SEED", "BENCH_SPLIT_STRATIFY", "BENCH_TRAIN_STEM", "BENCH_SPEED_TOKEN")


class SkippingSampler(torch.utils.data.Sampler[int]):
    def __init__(self, sampler: torch.utils.data.DistributedSampler | torch.utils.data.SequentialSampler) -> None:
        self.sampler = sampler
        self.epoch = 0
        self.skip_epoch = -1
        self.skip = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if isinstance(self.sampler, torch.utils.data.DistributedSampler):
            self.sampler.set_epoch(epoch)

    def skip_in_epoch(self, epoch: int, n_indices: int) -> None:
        self.skip_epoch, self.skip = epoch, n_indices

    def __len__(self) -> int:
        return len(self.sampler)

    def __iter__(self) -> Iterator[int]:
        skip = self.skip if self.epoch == self.skip_epoch else 0
        return itertools.islice(iter(self.sampler), skip, None)


def episode_digest(episodes: Sequence[int] | None) -> str:
    return hashlib.sha256(
        json.dumps(None if episodes is None else sorted(int(e) for e in episodes)).encode()
    ).hexdigest()


def data_env() -> tuple[tuple[str, str], ...]:
    return tuple((key, os.environ.get(key, "")) for key in _DATA_ENV)


class RunSetup(NamedTuple):
    dataset: str
    batch_size: int
    n_epochs: int
    n_batches: int
    lr: float
    lr_scheduler: str
    n_warmup_steps: int
    test_freq_batch: int
    train_episodes: str
    test_episodes: str
    data_env: tuple[tuple[str, str], ...]


class RngState(TypedDict):
    python: tuple[object, ...]
    numpy: tuple[str, np.ndarray, int, int, float] | dict[str, object]
    torch: torch.Tensor
    cuda: NotRequired[list[torch.Tensor]]


class TrainingState(TypedDict):
    setup: list[object]
    epoch: int
    batch: int
    n_itr: int
    policy: dict[str, torch.Tensor]
    optimizer: dict[str, object]
    lr_scheduler: dict[str, object]
    train_losses: dict[str, object]
    wandb_run_id: str | None
    test_iterator_built: bool
    rng: RngState


class Resumed(NamedTuple):
    epoch: int
    batch: int
    train_losses: dict[str, object]
    wandb_run_id: str | None
    test_iterator_built: bool
    rng: RngState


def _capture_rng() -> RngState:
    rng: RngState = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    return rng


def _restore_rng(rng: RngState) -> None:
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if "cuda" in rng:
        torch.cuda.set_rng_state_all(rng["cuda"])


def _unwrap(policy: torch.nn.Module) -> torch.nn.Module:
    module = getattr(policy, "module", policy)
    assert isinstance(module, torch.nn.Module)
    return module


def _as_record(setup: RunSetup) -> list[object]:
    return json.loads(json.dumps(setup))


class Resumable:
    def __init__(self, every: int, resume: bool) -> None:
        self.every = every
        self.resume = resume
        self.sampler: SkippingSampler | None = None
        self.path: Path | None = None
        self.setup: RunSetup | None = None

    @classmethod
    def from_env(cls, grad_accum_steps: int) -> Resumable | None:
        every = int(os.environ.get("BENCH_STATE_EVERY") or 0)
        resume = os.environ.get("BENCH_RESUME", "0") == "1"
        if every < 0:
            raise ValueError(f"BENCH_STATE_EVERY must be >= 0, got {every}")
        if not every and not resume:
            return None
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size != 1:
            raise RuntimeError(f"resumable training supports one process; WORLD_SIZE is {world_size}")
        if grad_accum_steps != 1:
            raise RuntimeError(f"resumable training needs GRAD_ACCUM_STEPS 1, got {grad_accum_steps}")
        return cls(every, resume)

    def wrap_sampler(self, sampler: torch.utils.data.DistributedSampler | None, dataset: Sized) -> SkippingSampler:
        self.sampler = SkippingSampler(sampler or torch.utils.data.SequentialSampler(dataset))
        return self.sampler

    def load(
        self,
        checkpoints_dir: str,
        setup: RunSetup,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> Resumed | None:
        self.path = Path(checkpoints_dir) / STATE_FILE
        self.setup = setup
        if not self.resume:
            return None
        if not self.path.is_file():
            logging.info("[bench] no training state at %s; starting from the beginning", self.path)
            return None
        state = cast("TrainingState", torch.load(self.path, map_location="cpu", weights_only=False))
        if state["setup"] != _as_record(setup):
            raise RuntimeError(
                f"{self.path} was saved by a different run setup: {state['setup']} != {_as_record(setup)}. "
                "Move it aside to start this configuration from the beginning."
            )
        _unwrap(policy).load_state_dict(state["policy"])
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        logging.info(
            "[bench] resuming from %s: epoch %d, batch %d, %d iterations done",
            self.path,
            state["epoch"] + 1,
            state["batch"],
            state["n_itr"],
        )
        return Resumed(
            state["epoch"],
            state["batch"],
            state["train_losses"],
            state["wandb_run_id"],
            state["test_iterator_built"],
            state["rng"],
        )

    def enter_epoch(self, resumed: Resumed, epoch: int) -> None:
        assert self.sampler is not None and self.setup is not None
        self.sampler.skip_in_epoch(epoch, resumed.batch * self.setup.batch_size)

    @staticmethod
    def restore_rng(resumed: Resumed, test_data_loader: torch.utils.data.DataLoader) -> None:
        if resumed.test_iterator_built:
            iter(test_data_loader)
        _restore_rng(resumed.rng)

    def maybe_save(
        self,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        *,
        epoch: int,
        batch_idx: int,
        n_itr: int,
        train_losses: object,
        wandb_run_id: str | None,
    ) -> None:
        if not self.every or (n_itr + 1) % self.every:
            return
        assert self.path is not None and self.setup is not None
        test_freq = self.setup.test_freq_batch
        state: TrainingState = {
            "setup": _as_record(self.setup),
            "epoch": epoch,
            "batch": batch_idx + 1,
            "n_itr": n_itr + 1,
            "policy": _unwrap(policy).state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "train_losses": dict(vars(train_losses)),
            "wandb_run_id": wandb_run_id,
            "test_iterator_built": epoch > 0 or bool(test_freq and n_itr >= test_freq),
            "rng": _capture_rng(),
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.path)
        logging.info("[bench] saved training state to %s (epoch %d, batch %d)", self.path, epoch + 1, batch_idx + 1)
