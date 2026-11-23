from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

EULER_RX_WRAPPED = "rx_wrapped"
EULER_XZ_WRAPPED = "xz_wrapped"
EULER_CONVENTIONS = (EULER_RX_WRAPPED, EULER_XZ_WRAPPED)

INCOMPLETE_MARKER = "bench_convert_incomplete.json"


def validate_convention(convention: str, *, source: str) -> str:
    if convention not in EULER_CONVENTIONS:
        raise RuntimeError(
            f"{source} declares euler_convention={convention!r}, which this pipeline does not implement "
            f"(known: {', '.join(EULER_CONVENTIONS)}). Serving would reject it at connection time, after "
            f"the whole run had been trained; refusing now instead."
        )
    return convention


FAMILY_RANGES = "ranges.json"


def family_ranges_path(root: Path) -> Path | None:
    for candidate in (root / "meta" / FAMILY_RANGES, root / FAMILY_RANGES):
        if candidate.is_file():
            return candidate
    return None


def parse_family_ranges(text: str, total_episodes: int) -> dict[str, tuple[int, int]]:
    entries = json.loads(text)
    try:
        ranges = {str(e["gym_id"]): (int(e["episodes"][0]), int(e["episodes"][1])) for e in entries}
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed entries ({exc})") from exc
    if not ranges:
        raise ValueError("no entries")
    if len(ranges) != len(entries):
        raise ValueError(f"duplicate gym_id: {len(entries)} entries, {len(ranges)} distinct ids")

    spans = sorted(ranges.values())
    if any(lo > hi for lo, hi in spans):
        raise ValueError(f"inverted span in {spans}")
    if spans[0][0] != 0:
        raise ValueError(f"starts at episode {spans[0][0]}, not 0")
    if spans[-1][1] != total_episodes - 1:
        raise ValueError(f"ends at episode {spans[-1][1]}, but the dataset has {total_episodes}")
    for a, b in zip(spans, spans[1:], strict=False):
        if a[1] + 1 != b[0]:
            kind = "overlap" if b[0] <= a[1] else "gap"
            raise ValueError(f"{kind} between {a} and {b}")
    return ranges


def mark_incomplete(root: Path, *, source: str) -> None:
    marker = root / "meta" / INCOMPLETE_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "started": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "reason": (
                    "conversion in progress; episodes already written use the rx_wrapped euler branch "
                    "while the euler marker that says so is not committed until every episode lands"
                ),
            },
            indent=2,
        )
    )


def clear_incomplete(root: Path) -> None:
    (root / "meta" / INCOMPLETE_MARKER).unlink(missing_ok=True)


def check_complete(root: Path) -> None:
    marker = root / "meta" / INCOMPLETE_MARKER
    if not marker.is_file():
        return
    try:
        started = json.loads(marker.read_text()).get("started", "at an unrecorded time")
    except (json.JSONDecodeError, OSError):
        started = "at an unrecorded time (the marker itself is unreadable)"
    raise RuntimeError(
        f"{root} was left mid-conversion ({marker} is present, written {started}). Its episodes already use "
        f"the rx_wrapped euler branch but its convention marker was never committed, so training would read "
        f"it as an xz_wrapped dataset (upstream's DOM convention) and serve the resulting checkpoint on the "
        f"wrong branch. Re-run the conversion to completion."
    )


def write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
