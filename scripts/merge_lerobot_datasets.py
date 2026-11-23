"""Merge per-family LeRobot datasets into one, keeping every family identifiable."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_GLOBAL_COLUMNS = ("episode_index", "index", "task_index")
_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_SHARD_KEYS = ("scene_style", "scene_idx", "workspace_offset", "main_seed")


class StatEntry(TypedDict):
    min: list[float]
    max: list[float]
    mean: list[float]
    std: list[float]
    count: list[int]


class EpisodeRow(TypedDict):
    episode_index: int
    tasks: list[str]
    length: int


class StatsRow(TypedDict):
    episode_index: int
    stats: dict[str, StatEntry]


class TaskRow(TypedDict):
    task_index: int
    task: str


class _ProvenanceRequired(TypedDict):
    episode_index: int


class ProvenanceRow(_ProvenanceRequired, total=False):
    gym_id: str


@dataclass(frozen=True)
class FamilyRange:
    gym_id: str
    first_episode: int
    last_episode: int
    episodes: int
    frames: int


@dataclass(frozen=True)
class Source:
    path: Path
    gym_id: str
    info: dict[str, object]
    stamp: dict[str, object] | None
    tasks: dict[int, str]
    episodes: dict[int, EpisodeRow]
    stats: dict[int, dict[str, StatEntry]]
    provenance: dict[int, ProvenanceRow] | None
    parquets: list[Path]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}:{lineno} is a {type(parsed).__name__}, not a record")
        rows.append(parsed)
    return rows


def _req_int(row: dict[str, object], key: str, where: str) -> int:
    value = row.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{where}: {key!r} is {value!r}, expected an int")
    return value


def _req_str(row: dict[str, object], key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{where}: {key!r} is {value!r}, expected a string")
    return value


def _features(info: dict[str, object], where: str) -> dict[str, dict[str, object]]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"{where}: 'features' is {type(features).__name__}, not a record")
    out: dict[str, dict[str, object]] = {}
    for name, spec in cast("dict[str, object]", features).items():
        if not isinstance(spec, dict):
            raise ValueError(f"{where}: feature {name!r} is {type(spec).__name__}, not a record")
        out[name] = cast("dict[str, object]", spec)
    return out


def _stamp(info: dict[str, object], where: str) -> dict[str, object] | None:
    if "bench_collection" not in info:
        return None
    stamp = info["bench_collection"]
    if not isinstance(stamp, dict):
        raise ValueError(f"{where}: bench_collection is {type(stamp).__name__}, not a record")
    record = cast("dict[str, object]", stamp)
    task_id = record.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(
            f"{where}: bench_collection is present but its task_id is {task_id!r} -- a damaged stamp "
            f"is not an absent one, and the family it names is what ranges.json is keyed on"
        )
    return record


def _stat_entry(value: object, where: str) -> StatEntry:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: statistics are {type(value).__name__}, not a record")
    entries = cast("dict[str, object]", value)
    out: dict[str, object] = {}
    for key in ("min", "max", "mean", "std", "count"):
        item = entries.get(key)
        if not isinstance(item, list) or not item:
            raise ValueError(f"{where}: statistic {key!r} is {item!r}, expected a non-empty list")
        out[key] = item
    return cast("StatEntry", out)


def _indexed(path: Path, indices: list[int], limit: int | None) -> None:
    repeated = sorted({i for i in indices if indices.count(i) > 1})
    if repeated:
        raise ValueError(f"{path}: episode_index repeated {repeated[:5]}")
    if limit is not None:
        beyond = sorted(i for i in indices if not 0 <= i < limit)
        if beyond:
            raise ValueError(f"{path}: episode_index {beyond[:5]} outside 0..{limit - 1}")


def _episode_rows(path: Path, limit: int | None = None) -> dict[int, EpisodeRow]:
    rows: dict[int, EpisodeRow] = {}
    seen: list[int] = []
    for row in _read_jsonl(path):
        index = _req_int(row, "episode_index", str(path))
        seen.append(index)
        tasks = row.get("tasks")
        if not isinstance(tasks, list) or not all(isinstance(t, str) for t in tasks):
            raise ValueError(f"{path}: episode {index} has tasks={tasks!r}, expected a list of strings")
        rows[index] = {
            "episode_index": index,
            "tasks": [str(t) for t in tasks],
            "length": _req_int(row, "length", f"{path} episode {index}"),
        }
    _indexed(path, seen, limit)
    return rows


def _stats_rows(path: Path, limit: int | None = None) -> dict[int, dict[str, StatEntry]]:
    rows: dict[int, dict[str, StatEntry]] = {}
    seen: list[int] = []
    for row in _read_jsonl(path):
        index = _req_int(row, "episode_index", str(path))
        seen.append(index)
        stats = row.get("stats")
        if not isinstance(stats, dict):
            raise ValueError(f"{path}: episode {index} has no stats record")
        columns = cast("dict[str, object]", stats)
        rows[index] = {
            name: _stat_entry(entry, f"{path} episode {index} column {name!r}") for name, entry in columns.items()
        }
    _indexed(path, seen, limit)
    return rows


def _task_rows(path: Path) -> dict[int, str]:
    return {_req_int(row, "task_index", str(path)): _req_str(row, "task", str(path)) for row in _read_jsonl(path)}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _episode_parquets(root: Path) -> list[Path]:
    numbered: list[tuple[int, Path]] = []
    for path in root.glob("data/*/*.parquet"):
        tail = path.stem.split("_")[-1]
        if not tail.isdigit():
            raise ValueError(f"{path}: episode file name does not end in an index")
        numbered.append((int(tail), path))
    numbered.sort()
    indices = [i for i, _ in numbered]
    if indices != list(range(len(numbered))):
        missing = sorted(set(range(len(numbered))) - set(indices))
        raise ValueError(
            f"{root}: episode files are not 0..{len(numbered) - 1}; missing {missing[:5]}, "
            f"highest {indices[-1] if indices else '-'}"
        )
    return [path for _, path in numbered]


def _check_shard(first: Source, stamp: dict[str, object] | None, path: Path) -> None:
    if first.stamp is None or stamp is None:
        raise ValueError(f"{path}: a shard must carry its collection stamp, which names its background and seed")
    differ = sorted(
        k for k in set(first.stamp) | set(stamp) if k not in _SHARD_KEYS and first.stamp.get(k) != stamp.get(k)
    )
    if differ:
        raise ValueError(f"{path}: stamp differs from {first.path}'s shard of {first.gym_id!r} in {differ}")
    shared = _seed_set(stamp, path) & _seed_set(first.stamp, first.path)
    if shared:
        raise ValueError(f"{path}: main_seed shares {sorted(shared)} with {first.path}, so they drew the same episodes")


def _seed_set(stamp: dict[str, object], where: Path) -> set[int]:
    seeds = stamp.get("main_seed")
    values = [s for s in seeds if isinstance(s, int) and not isinstance(s, bool)] if isinstance(seeds, list) else []
    if not values or len(values) != len(cast("list[object]", seeds)):
        raise ValueError(f"{where}: a shard's main_seed must be a non-empty list of ints, got {seeds!r}")
    return set(values)


def read_sources(paths: list[Path], *, allow_shards: bool = False) -> list[Source]:
    sources: list[Source] = []
    template: dict[str, object] | None = None
    seen: dict[str, Path] = {}
    for path in paths:
        info_path = path / "meta" / "info.json"
        if not info_path.is_file():
            raise ValueError(f"{path}: no meta/info.json, so this is not a LeRobot dataset")
        parsed = json.loads(info_path.read_text())
        if not isinstance(parsed, dict):
            raise ValueError(f"{info_path} is a {type(parsed).__name__}, not a record")
        info = cast("dict[str, object]", parsed)
        if template is None:
            template = info
        _req_int(info, "chunks_size", str(info_path))
        for key in ("codebase_version", "robot_type", "fps", "features", "chunks_size"):
            if info.get(key) != template.get(key):
                raise ValueError(f"{path}: {key} differs from the first source, so these are not one dataset")
        if info.get("bench_schema") != template.get("bench_schema"):
            raise ValueError(f"{path}: bench_schema differs -- two camera vintages cannot merge")
        video = sorted(k for k, v in _features(info, str(info_path)).items() if v.get("dtype") == "video")
        if video:
            raise ValueError(
                f"{path}: declares video feature(s) {video}, which this merge does not carry -- the "
                f"result would name video files that do not exist"
            )
        stamp = _stamp(info, str(info_path))
        gym_id = str(stamp["task_id"]) if stamp is not None else path.name
        if gym_id in seen:
            if not allow_shards:
                raise ValueError(
                    f"{path}: gym id {gym_id!r} is already claimed by {seen[gym_id]} -- merge shards of one "
                    f"family into a single source first, or ranges.json cannot address them apart"
                )
            if sources[-1].gym_id != gym_id:
                raise ValueError(f"{path}: shards of {gym_id!r} are not adjacent ({seen[gym_id]} came earlier)")
            for earlier in sources:
                if earlier.gym_id == gym_id:
                    _check_shard(earlier, stamp, path)
        seen.setdefault(gym_id, path)
        parquets = _episode_parquets(path)
        if not parquets:
            raise ValueError(f"{path}: no episode files, so it would contribute an empty, inverted range")
        declared = _req_int(info, "total_episodes", str(info_path))
        if len(parquets) != declared:
            raise ValueError(
                f"{path}: {len(parquets)} episode files but info.json declares {declared} -- "
                f"the source is incomplete, and a contiguous 0..{len(parquets) - 1} hides it"
            )
        prov_path = path / "meta" / "bench_episode_provenance.jsonl"
        provenance = None
        if prov_path.is_file():
            raw = [(_req_int(r, "episode_index", str(prov_path)), r) for r in _read_jsonl(prov_path)]
            _indexed(prov_path, [i for i, _ in raw], len(parquets))
            rows: dict[int, ProvenanceRow] = {i: cast("ProvenanceRow", r) for i, r in raw}
            missing = [i for i in range(len(parquets)) if i not in rows]
            if missing:
                raise ValueError(
                    f"{path}: provenance covers {len(rows)} of {len(parquets)} episodes (missing "
                    f"{missing[:5]}) -- a partial sidecar reads as complete provenance"
                )
            provenance = rows
        episodes = _episode_rows(path / "meta" / "episodes.jsonl", declared)
        stats = _stats_rows(path / "meta" / "episodes_stats.jsonl", declared)
        for name, covered in (("episodes.jsonl", episodes), ("episodes_stats.jsonl", stats)):
            absent = [i for i in range(declared) if i not in covered]
            if absent:
                raise ValueError(f"{path}: {name} covers {len(covered)} of {declared} episodes (missing {absent[:5]})")
        sources.append(
            Source(
                path=path,
                gym_id=gym_id,
                info=info,
                stamp=stamp,
                tasks=_task_rows(path / "meta" / "tasks.jsonl"),
                episodes=episodes,
                stats=stats,
                provenance=provenance,
                parquets=parquets,
            )
        )
    carried = [s.provenance is not None for s in sources]
    if any(carried) and not all(carried):
        without = [s.path.name for s in sources if s.provenance is None]
        raise ValueError(f"some sources carry an episode-provenance sidecar and these do not: {without}")
    return sources


def _payload_digest(table: pa.Table) -> str:
    h = hashlib.sha256()
    for name in table.schema.names:
        if name in _GLOBAL_COLUMNS:
            continue
        for value in table.column(name):
            item = value.as_py()
            payload = item["bytes"] if isinstance(item, dict) and "bytes" in item else repr(item).encode()
            h.update(payload if isinstance(payload, bytes) else bytes(payload))
    return h.hexdigest()


def _renumber(table: pa.Table, *, episode: int, frame_offset: int, task_map: dict[int, int]) -> pa.Table:
    rows = table.num_rows
    source_tasks = table.column("task_index").to_pylist()
    replacements = {
        "episode_index": pa.array([episode] * rows, type=table.schema.field("episode_index").type),
        "index": pa.array(range(frame_offset, frame_offset + rows), type=table.schema.field("index").type),
        "task_index": pa.array([task_map[int(t)] for t in source_tasks], type=table.schema.field("task_index").type),
    }
    for name, values in replacements.items():
        table = table.set_column(table.schema.get_field_index(name), name, values)
    return table


def _point_stat(value: int, count: list[int]) -> StatEntry:
    return {"min": [value], "max": [value], "mean": [float(value)], "std": [0.0], "count": count}


def _column_stat(values: list[int], count: list[int]) -> StatEntry:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {"min": [min(values)], "max": [max(values)], "mean": [mean], "std": [var**0.5], "count": count}


def _move_stats(
    stats: dict[str, StatEntry],
    *,
    episode: int,
    task_values: list[int],
    frame_offset: int,
    old_frame_offset: int,
) -> dict[str, StatEntry]:
    out = dict(stats)
    if "episode_index" in out:
        out["episode_index"] = _point_stat(episode, out["episode_index"]["count"])
    if "task_index" in out:
        out["task_index"] = _column_stat(task_values, out["task_index"]["count"])
    if "index" in out:
        source = out["index"]
        delta = frame_offset - old_frame_offset
        out["index"] = {
            "min": [v + delta for v in source["min"]],
            "max": [v + delta for v in source["max"]],
            "mean": [v + delta for v in source["mean"]],
            "std": source["std"],
            "count": source["count"],
        }
    return out


def _refuse_overlap(sources: list[Source], out: Path) -> None:
    resolved_out = out.resolve()
    for source in sources:
        resolved = source.path.resolve()
        if resolved == resolved_out or resolved_out in resolved.parents or resolved in resolved_out.parents:
            raise ValueError(
                f"--out {out} overlaps source {source.path}: the merge deletes the destination before "
                f"writing, which would delete that source first"
            )


def merge(source_paths: list[Path], out: Path, *, verify_sample: int, allow_shards: bool = False) -> list[FamilyRange]:
    sources = read_sources(source_paths, allow_shards=allow_shards)
    _refuse_overlap(sources, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    try:
        ranges = _merge_into(sources, stage, verify_sample=verify_sample, tag_background=allow_shards)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    previous = out.with_name(f".{out.name}.replaced-{os.getpid()}") if out.exists() else None
    try:
        if previous is not None:
            out.rename(previous)
        try:
            stage.rename(out)
        except BaseException:
            if previous is not None:
                previous.rename(out)
            raise
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    if previous is not None:
        shutil.rmtree(previous, ignore_errors=True)
    return ranges


def _merge_into(
    sources: list[Source], out: Path, *, verify_sample: int, tag_background: bool = False
) -> list[FamilyRange]:
    (out / "meta").mkdir(parents=True)

    template = sources[0].info
    chunk_size = _req_int(template, "chunks_size", str(sources[0].path))
    task_index_of: dict[str, int] = {}
    tasks: list[TaskRow] = []
    episodes: list[EpisodeRow] = []
    stats: list[StatsRow] = []
    provenance: list[dict[str, object]] = []
    ranges: list[FamilyRange] = []
    episode = 0
    frame = 0
    payload_schema: pa.Schema | None = None

    for source in sources:
        first_episode, first_frame = episode, frame
        sample_every = max(1, len(source.parquets) // verify_sample) if verify_sample else 0

        for local, path in enumerate(source.parquets):
            table = pq.read_table(path)
            if payload_schema is None:
                absent = sorted(set(_features(source.info, str(source.path))) - set(table.schema.names))
                if absent:
                    raise ValueError(f"{path}: info.json declares feature(s) {absent} that the parquet does not carry")
                payload_schema = table.schema
            elif not table.schema.equals(payload_schema):
                raise ValueError(
                    f"{path}: parquet schema differs from the first source's. "
                    f"here {table.schema.names} / {[str(t) for t in table.schema.types]}, "
                    f"expected {payload_schema.names} / {[str(t) for t in payload_schema.types]}"
                )
            task_map: dict[int, int] = {}
            for raw in dict.fromkeys(int(t) for t in table.column("task_index").to_pylist()):
                if raw not in source.tasks:
                    raise ValueError(f"{path}: task_index {raw} is not in the source's tasks.jsonl")
                prompt = source.tasks[raw]
                if prompt not in task_index_of:
                    task_index_of[prompt] = len(task_index_of)
                    tasks.append({"task_index": task_index_of[prompt], "task": prompt})
                task_map[raw] = task_index_of[prompt]
            old_frame_offset = int(table.column("index")[0].as_py())
            digest = _payload_digest(table) if sample_every and local % sample_every == 0 else None

            declared_length = source.episodes[local]["length"]
            if table.num_rows != declared_length:
                raise ValueError(
                    f"{path}: {table.num_rows} rows but episodes.jsonl says {declared_length} -- the "
                    f"merged dataset would carry a length its own table contradicts"
                )
            merged = _renumber(table, episode=episode, frame_offset=frame, task_map=task_map)
            task_values = [int(t) for t in merged.column("task_index").to_pylist()]
            target = out / _DATA_PATH.format(episode_chunk=episode // chunk_size, episode_index=episode)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(merged, target)
            if digest is not None and _payload_digest(pq.read_table(target)) != digest:
                raise ValueError(f"{path} -> {target}: payload changed during the merge")

            episodes.append(
                {
                    "episode_index": episode,
                    "tasks": source.episodes[local]["tasks"],
                    "length": source.episodes[local]["length"],
                }
            )
            stats.append(
                {
                    "episode_index": episode,
                    "stats": _move_stats(
                        source.stats[local],
                        episode=episode,
                        task_values=task_values,
                        frame_offset=frame,
                        old_frame_offset=old_frame_offset,
                    ),
                }
            )
            if source.provenance is not None:
                row = {
                    **cast("dict[str, object]", source.provenance[local]),
                    "episode_index": episode,
                    "gym_id": source.gym_id,
                }
                if tag_background and source.stamp is not None:
                    stamp_tags = {k: source.stamp.get(k) for k in ("scene_style", "scene_idx", "workspace_offset")}
                    row = stamp_tags | row
                provenance.append(row)

            episode += 1
            frame += merged.num_rows

        ranges.append(
            FamilyRange(source.gym_id, first_episode, episode - 1, episode - first_episode, frame - first_frame)
        )

    info = dict(template)
    info.update(
        {
            "total_episodes": episode,
            "total_frames": frame,
            "total_tasks": len(tasks),
            "total_videos": 0,
            "total_chunks": (episode + chunk_size - 1) // chunk_size,
            "splits": {"train": f"0:{episode}"},
            "data_path": _DATA_PATH,
        }
    )
    info.pop("bench_collection", None)
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    _write_jsonl(out / "meta" / "tasks.jsonl", tasks)
    _write_jsonl(out / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(out / "meta" / "episodes_stats.jsonl", stats)
    if provenance:
        _write_jsonl(out / "meta" / "bench_episode_provenance.jsonl", provenance)

    manifest: list[dict[str, object]] = []
    for r, source in zip(ranges, sources, strict=True):
        stamp = source.stamp or {}
        previous = manifest[-1] if manifest else None
        if previous is None or previous["gym_id"] != r.gym_id:
            manifest.append(
                {
                    "gym_id": r.gym_id,
                    "episodes": [r.first_episode, r.last_episode],
                    "count": r.episodes,
                    "frames": r.frames,
                    "collection": stamp,
                }
            )
            continue
        if "shards" not in previous:
            first = cast("dict[str, object]", previous["collection"])
            previous["shards"] = [
                {"episodes": previous["episodes"], "count": previous["count"], "frames": previous["frames"]}
                | {k: first.get(k) for k in _SHARD_KEYS}
            ]
            previous["collection"] = {k: v for k, v in first.items() if k not in _SHARD_KEYS}
        cast("list[dict[str, object]]", previous["shards"]).append(
            {"episodes": [r.first_episode, r.last_episode], "count": r.episodes, "frames": r.frames}
            | {k: stamp.get(k) for k in _SHARD_KEYS}
        )
        previous["episodes"] = [cast("list[int]", previous["episodes"])[0], r.last_episode]
        previous["count"] = cast("int", previous["count"]) + r.episodes
        previous["frames"] = cast("int", previous["frames"]) + r.frames
    (out / "meta" / "ranges.json").write_text(json.dumps(manifest, indent=1))
    (out / "ranges.json").write_text(
        json.dumps([{k: m[k] for k in ("gym_id", "episodes")} for m in manifest], indent=1)
    )
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sources", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--allow-shards",
        action="store_true",
        help="merge consecutive sources of one family (one run per background) into one range; see merge()",
    )
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=5,
        help="episodes per source whose non-renumbered columns are hashed before and after the rewrite (0 disables)",
    )
    args = parser.parse_args()

    ranges = merge(sorted(args.sources), args.out, verify_sample=args.verify_sample, allow_shards=args.allow_shards)
    total_e = sum(r.episodes for r in ranges)
    total_f = sum(r.frames for r in ranges)
    print(f"merged {len(ranges)} sources -> {total_e} episodes / {total_f} frames at {args.out}")
    for r in ranges[:5]:
        print(f"  {r.gym_id}: {r.episodes} eps, {r.first_episode}..{r.last_episode}")
    if len(ranges) > 5:
        print(f"  ... and {len(ranges) - 5} more")


if __name__ == "__main__":
    main()
