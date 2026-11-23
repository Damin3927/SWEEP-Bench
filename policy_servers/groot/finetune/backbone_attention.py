from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib

RECORD = "bench_backbone_attention.jsonl"
MODES = ("sdpa",)


def from_env() -> str | None:
    mode = os.environ.get("BENCH_GROOT_ATTENTION") or None
    if mode is not None and mode not in MODES:
        raise SystemExit(f"BENCH_GROOT_ATTENTION={mode!r} is not one of {MODES}")
    return mode


def install(mode: str, run_dir: pathlib.Path, existing_steps: tuple[int, ...]) -> None:
    import gr00t.model.gr00t_n1d7.gr00t_n1d7 as gr00t_n1d7

    original = gr00t_n1d7.Gr00tN1d7.__init__

    def init(self, config, *args, **kwargs):
        config.use_flash_attention = False
        original(self, config, *args, **kwargs)

    gr00t_n1d7.Gr00tN1d7.__init__ = init
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / RECORD).open("a") as f:
        f.write(json.dumps({"attention": mode, "from_step": max(existing_steps, default=0)}) + "\n")
