from __future__ import annotations

import json
import os
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

ENV_VAR = "BENCH_EXTERIOR_HISTORY"
CHECKPOINT_STAMP = "bench_exterior_history.json"
METADATA_KEY = "exterior_history"
_STAMP_FIELD = "exterior_history"


def validate_offsets(offsets: Iterable[int]) -> tuple[int, ...]:
    result = tuple(offsets)
    if any(not isinstance(o, int) or isinstance(o, bool) for o in result):
        raise ValueError(f"exterior history offsets must be ints, got {result!r}")
    if any(o >= 0 for o in result) or list(result) != sorted(set(result)):
        raise ValueError(f"exterior history offsets must be strictly increasing negative ints, got {result!r}")
    return result


def parse_offsets(raw: str | None) -> tuple[int, ...]:
    if raw is None or not raw.strip():
        return ()
    return validate_offsets(int(token) for token in raw.split(","))


def offsets_from_env() -> tuple[int, ...]:
    return parse_offsets(os.environ.get(ENV_VAR))


def image_key(offset: int) -> str:
    return f"base_0_rgb_tm{-offset:02d}"


def stamp_run_dir(run_dir: pathlib.Path, offsets: tuple[int, ...], *, existing_steps: tuple[int, ...]) -> None:
    path = pathlib.Path(run_dir) / CHECKPOINT_STAMP
    if path.is_file():
        recorded = read_stamp(path)
        if recorded != offsets:
            raise RuntimeError(
                f"{run_dir} holds checkpoints trained with exterior history {recorded}, but this run "
                f"uses {offsets}. Use a fresh --exp-name."
            )
        return
    if existing_steps and offsets:
        raise RuntimeError(
            f"{run_dir} already holds checkpoints {existing_steps} without {CHECKPOINT_STAMP}, so they "
            f"were trained single-frame; resuming them with exterior history {offsets} would mix the two. "
            f"Use a fresh --exp-name."
        )
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({_STAMP_FIELD: list(offsets)}))
    tmp.replace(path)
    print(f"[bench] stamped {path} with exterior history {offsets}", flush=True)


def read_stamp(path: pathlib.Path) -> tuple[int, ...]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get(_STAMP_FIELD), list):
        raise ValueError(f"{path} is not a valid {CHECKPOINT_STAMP}: {data!r}")
    return validate_offsets(data[_STAMP_FIELD])


def checkpoint_offsets(ckpt_dir: str | pathlib.Path) -> tuple[int, ...] | None:
    path = pathlib.Path(ckpt_dir)
    for candidate in (path, *path.parents):
        stamp = candidate / CHECKPOINT_STAMP
        if stamp.is_file():
            return read_stamp(stamp)
    return None


def check_serve(ckpt_dir: str | pathlib.Path, configured: tuple[int, ...]) -> None:
    recorded = checkpoint_offsets(ckpt_dir)
    trained = () if recorded is None else recorded
    if trained != configured:
        source = "it has no stamp, so it is single-frame" if recorded is None else f"per its {CHECKPOINT_STAMP}"
        raise RuntimeError(
            f"checkpoint {ckpt_dir} was trained with exterior history {trained} ({source}), but "
            f"{ENV_VAR} configures {configured}. Serve it with {ENV_VAR}="
            f"{','.join(str(o) for o in trained)!r}."
        )
