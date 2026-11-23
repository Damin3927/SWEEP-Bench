from __future__ import annotations

import json
import os
import pathlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

ENV_VAR = "BENCH_EXTERIOR_HISTORY"
CHECKPOINT_STAMP = "bench_exterior_history.json"
METADATA_KEY = "exterior_history"
EXTERIOR_COLUMN = "exterior_image_1_left"
_STAMP_FIELD = "exterior_history"
_KEY_PATTERN = re.compile(rf"^{EXTERIOR_COLUMN}_tm(\d{{2,}})$")


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


def video_key(offset: int) -> str:
    validate_offsets((offset,))
    return f"{EXTERIOR_COLUMN}_tm{-offset:02d}"


def parse_video_key(key: str) -> tuple[str, int] | None:
    match = _KEY_PATTERN.match(key)
    if match is None:
        return None
    return EXTERIOR_COLUMN, -int(match.group(1))


def stamp_run_dir(run_dir: pathlib.Path, offsets: tuple[int, ...], *, existing_checkpoints: tuple[str, ...]) -> None:
    path = pathlib.Path(run_dir) / CHECKPOINT_STAMP
    if path.is_file():
        recorded = read_stamp(path)
        if recorded != offsets:
            raise RuntimeError(
                f"{run_dir} holds checkpoints trained with exterior history {recorded}, but this run "
                f"uses {offsets}. Use a fresh --output-dir."
            )
        return
    if existing_checkpoints and offsets:
        raise RuntimeError(
            f"{run_dir} already holds {existing_checkpoints} without {CHECKPOINT_STAMP}, so they were "
            f"trained single-frame; resuming them with exterior history {offsets} would mix the two. "
            f"Use a fresh --output-dir."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if recorded is None:
        raise RuntimeError(f"checkpoint {ckpt_dir} has no {CHECKPOINT_STAMP} in it or any parent dir")
    if recorded != configured:
        raise RuntimeError(
            f"checkpoint {ckpt_dir} was trained with exterior history {recorded} (per its "
            f"{CHECKPOINT_STAMP}), but {ENV_VAR} configures {configured}. Serve it with "
            f"{ENV_VAR}={','.join(str(o) for o in recorded)!r}."
        )
