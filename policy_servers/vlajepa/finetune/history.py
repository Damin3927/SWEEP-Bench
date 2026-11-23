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


def clamped_indices(t: int, offsets: tuple[int, ...]) -> list[int]:
    return [max(0, t + o) for o in (*offsets, 0)]


def read_stamp(path: pathlib.Path) -> tuple[int, ...]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get(_STAMP_FIELD), list):
        raise ValueError(f"{path} is not a valid {CHECKPOINT_STAMP}: {data!r}")
    return validate_offsets(data[_STAMP_FIELD])


def stamp_run_dir(run_dir: pathlib.Path, offsets: tuple[int, ...], *, has_checkpoints: bool) -> None:
    path = pathlib.Path(run_dir) / CHECKPOINT_STAMP
    if path.is_file():
        recorded = read_stamp(path)
        if recorded != offsets:
            raise RuntimeError(
                f"{run_dir} holds checkpoints trained with exterior history {recorded}, but this run "
                f"uses {offsets}. Use a fresh run dir."
            )
        return
    if has_checkpoints:
        raise RuntimeError(f"{run_dir} holds checkpoints but no {CHECKPOINT_STAMP}; refusing to guess their history")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({_STAMP_FIELD: list(offsets)}))
    tmp.replace(path)


def checkpoint_offsets(ckpt: str | pathlib.Path) -> tuple[int, ...]:
    path = pathlib.Path(ckpt)
    for candidate in (path, *path.parents):
        stamp = candidate / CHECKPOINT_STAMP
        if stamp.is_file():
            return read_stamp(stamp)
    raise FileNotFoundError(f"no {CHECKPOINT_STAMP} at or above {ckpt}")


def check_serve(ckpt: str | pathlib.Path, configured: tuple[int, ...]) -> None:
    trained = checkpoint_offsets(ckpt)
    if trained != configured:
        raise RuntimeError(
            f"checkpoint {ckpt} was trained with exterior history {trained} (per its {CHECKPOINT_STAMP}), "
            f"but {ENV_VAR} configures {configured}. Serve it with {ENV_VAR}="
            f"{','.join(str(o) for o in trained)!r}."
        )
