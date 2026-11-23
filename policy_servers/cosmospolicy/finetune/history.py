from __future__ import annotations

import dataclasses
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
_SCHEMA_FIELD = "bench_schema"

FRAMES_PER_LATENT = 4


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


def clamped_frame_indices(t: int, offsets: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(max(0, t + o) for o in (*offsets, 0))


@dataclasses.dataclass(frozen=True)
class LatentLayout:
    history: tuple[int, ...]

    @property
    def current_proprio(self) -> int:
        return 1

    @property
    def current_wrist(self) -> int:
        return 2

    @property
    def exterior(self) -> tuple[int, ...]:
        return tuple(range(3, 3 + len(self.history) + 1))

    @property
    def current_exterior(self) -> int:
        return self.exterior[-1]

    @property
    def action(self) -> int:
        return self.current_exterior + 1

    @property
    def future_proprio(self) -> int:
        return self.action + 1

    @property
    def future_wrist(self) -> int:
        return self.action + 2

    @property
    def future_exterior(self) -> int:
        return self.action + 3

    @property
    def value(self) -> int:
        return self.action + 4

    @property
    def state_t(self) -> int:
        return self.value + 1

    @property
    def num_conditional_frames(self) -> int:
        return self.action

    @property
    def num_video_frames(self) -> int:
        return 1 + FRAMES_PER_LATENT * (self.state_t - 1)

    def latent_sources(self) -> tuple[str, ...]:
        return (
            "blank",
            "blank",
            "wrist",
            *(f"exterior@{o}" for o in (*self.history, 0)),
            "blank",
            "blank",
            "future_wrist",
            "future_exterior",
            "blank",
        )

    def frame_sources(self) -> tuple[str, ...]:
        sources = self.latent_sources()
        return (sources[0], *(s for s in sources[1:] for _ in range(FRAMES_PER_LATENT)))


def latent_of_frame(frame: int) -> int:
    return 0 if frame == 0 else 1 + (frame - 1) // FRAMES_PER_LATENT


def stamp_run_dir(run_dir: pathlib.Path, offsets: tuple[int, ...], bench_schema: str | None) -> None:
    path = pathlib.Path(run_dir) / CHECKPOINT_STAMP
    if path.is_file():
        recorded = read_stamp(path)
        if recorded != (offsets, bench_schema):
            raise RuntimeError(
                f"{run_dir} holds a run stamped (history, bench_schema)={recorded}, but this run uses "
                f"{(offsets, bench_schema)}. Use a fresh job name."
            )
        return
    checkpoints = pathlib.Path(run_dir) / "checkpoints"
    if checkpoints.is_dir() and any(checkpoints.iterdir()):
        raise RuntimeError(f"{run_dir} already holds checkpoints without {CHECKPOINT_STAMP}. Use a fresh job name.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({_STAMP_FIELD: list(offsets), _SCHEMA_FIELD: bench_schema}))
    tmp.replace(path)
    print(f"[bench] stamped {path} with exterior history {offsets}, bench_schema {bench_schema}", flush=True)


def read_stamp(path: pathlib.Path) -> tuple[tuple[int, ...], str | None]:
    data = json.loads(pathlib.Path(path).read_text())
    if not isinstance(data, dict) or not isinstance(data.get(_STAMP_FIELD), list):
        raise ValueError(f"{path} is not a valid {CHECKPOINT_STAMP}: {data!r}")
    schema = data.get(_SCHEMA_FIELD)
    if schema is not None and not isinstance(schema, str):
        raise ValueError(f"{path} has a non-string {_SCHEMA_FIELD}: {schema!r}")
    return validate_offsets(data[_STAMP_FIELD]), schema


def find_stamp(ckpt_dir: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(ckpt_dir)
    for candidate in (path, *path.parents):
        stamp = candidate / CHECKPOINT_STAMP
        if stamp.is_file():
            return stamp
    raise FileNotFoundError(
        f"no {CHECKPOINT_STAMP} above {ckpt_dir}; every bench Cosmos Policy run writes one, so this is not a "
        "bench checkpoint"
    )


def check_serve(ckpt_dir: str | pathlib.Path, configured: tuple[int, ...]) -> tuple[tuple[int, ...], str | None]:
    trained, schema = read_stamp(find_stamp(ckpt_dir))
    if trained != configured:
        raise RuntimeError(
            f"checkpoint {ckpt_dir} was trained with exterior history {trained}, but {ENV_VAR} configures "
            f"{configured}. Serve it with {ENV_VAR}={','.join(str(o) for o in trained)!r}."
        )
    return trained, schema
