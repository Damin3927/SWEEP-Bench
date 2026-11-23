"""Vet candidate objects into a task-ready manifest."""

from __future__ import annotations

import argparse
import functools
import logging
from pathlib import Path

from sweep_bench.assets.mesh_import import default_cache_dir
from sweep_bench.vetting import (
    _WORKER_OBJECT_TIMEOUT,
    MAX_ZERO_PROGRESS_CANDIDATES,
    VET_SPEED_DEFAULT,
    Source,
    VetResult,
    VetRunConfig,
    ZeroProgressAbort,
    append_result_jsonl,
    build_manifest_dict,
    candidate_stale_reason,
    check_results_config,
    default_vetted_manifest_path,
    discard_unused_results_claim,
    harness_identity,
    load_candidates,
    read_results_jsonl,
    run_isolated,
    run_vetting,
    write_manifest,
)

logger = logging.getLogger("vet_objects")


def _parse_shard(spec: str) -> tuple[int, int]:
    try:
        k_str, n_str = spec.split("/", 1)
        k, n = int(k_str), int(n_str)
    except ValueError as e:
        raise ValueError(f"--shard must look like K/N (got {spec!r})") from e
    if n < 1 or not (0 <= k < n):
        raise ValueError(f"--shard K/N needs 0 <= K < N and N >= 1 (got {spec!r})")
    return k, n


_ALL_SOURCES: list[Source] = ["procedural", "gso", "ycb", "robocasa", "polyhaven"]


def _resolve_sources(choice: str) -> list[Source]:
    if choice == "all":
        return list(_ALL_SOURCES)
    if choice == "procedural":
        return ["procedural"]
    if choice == "ycb":
        return ["ycb"]
    if choice == "robocasa":
        return ["robocasa"]
    if choice == "polyhaven":
        return ["polyhaven"]
    return ["gso"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sources", choices=("procedural", "gso", "ycb", "robocasa", "polyhaven", "all"), default="all"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Imported-mesh candidate manifest, shared by gso and ycb (default: <cache>/gso_manifest.json).",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Vetted manifest path (default: <cache>/vetted_manifest.json)."
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="GSO asset cache root (default: ~/.cache/sweep_bench/gso)."
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=VET_SPEED_DEFAULT,
        help="Belt speed (m/s) for the ride checks (vet band 0.10-0.30).",
    )
    parser.add_argument(
        "--sample-limit", type=int, default=None, help="Vet at most this many objects PER source (quick sample)."
    )
    parser.add_argument(
        "--only", nargs="+", default=None, help="Vet only these object ids (reproduce/target a single object)."
    )
    parser.add_argument(
        "--shard",
        default=None,
        metavar="K/N",
        help="Vet only candidates[K::N] (N parallel processes over one candidate list). "
        "The --out manifest is auto-suffixed per shard so parallel shards never write the same file.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Append-only JSONL of per-object verdicts, written as they land (survives a kill, merges shards).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip objects already recorded in --results (requires --results).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sim seed; the vet decision is stable for a fixed seed.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-object worker timeout (s) before a hung object is REJECTed as 'timeout'.",
    )
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="Vet in-process (no crash isolation). Faster, but ANY object that hard-crashes kills the run.",
    )
    parser.add_argument(
        "--max-zero-progress",
        type=int,
        default=MAX_ZERO_PROGRESS_CANDIDATES,
        help="Abort after this many CANDIDATES in a row vet nothing, with no worker vetting anything in "
        "between, instead of rejecting one candidate per restart. Raise it only when you have "
        f"established that they really are consecutive bad assets (default {MAX_ZERO_PROGRESS_CANDIDATES}).",
    )
    parser.add_argument("--robot", default="panda")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cache_dir: Path = args.cache_dir or default_cache_dir()
    out_path: Path = args.out or default_vetted_manifest_path(cache_dir)
    sources = _resolve_sources(args.sources)

    if args.resume and args.results is None:
        parser.error("--resume needs --results (the file it resumes from)")
    if args.timeout is not None and args.timeout <= 0:
        parser.error(f"--timeout must be positive, got {args.timeout}")
    if args.max_zero_progress < 1:
        parser.error(f"--max-zero-progress must be at least 1, got {args.max_zero_progress}")

    candidates = load_candidates(
        sources,
        gso_manifest_path=args.manifest,
        cache_dir=cache_dir,
        sample_limit=args.sample_limit,
        only=args.only,
    )
    if not candidates:
        parser.error("no candidates to vet (empty procedural catalog and/or missing GSO manifest)")

    remaining = list(candidates)
    if args.shard is not None:
        try:
            k, n = _parse_shard(args.shard)
        except ValueError as e:
            parser.error(str(e))
        remaining = remaining[k::n]
        out_path = out_path.with_name(f"{out_path.stem}.shard{k}of{n}{out_path.suffix}")
        logger.info("shard %d/%d: %d of %d candidates; manifest -> %s", k, n, len(remaining), len(candidates), out_path)

    owns_results_claim = False
    if args.results is not None:
        effective_timeout = (
            None if args.no_isolation else (_WORKER_OBJECT_TIMEOUT if args.timeout is None else args.timeout)
        )
        try:
            owns_results_claim = check_results_config(
                args.results,
                VetRunConfig(
                    speed=args.speed,
                    seed=args.seed,
                    robot_uids=args.robot,
                    object_timeout=effective_timeout,
                    harness=harness_identity(),
                ),
            )
        except ValueError as e:
            parser.error(str(e))

    recorded: dict[str, VetResult] = read_results_jsonl(args.results) if args.results is not None else {}
    on_result = None if args.results is None else functools.partial(append_result_jsonl, args.results)
    if args.resume:
        before = len(remaining)
        moved = {
            c.id: why
            for c in remaining
            if c.id in recorded and (why := candidate_stale_reason(recorded[c.id], c)) is not None
        }
        remaining = [c for c in remaining if c.id not in recorded or c.id in moved]
        if moved:
            logger.info("resume: %d recorded id(s) no longer describe the candidate; RE-VETTING them", len(moved))
            for object_id, why in moved.items():
                logger.info("  %s: %s", object_id, why)
        logger.info("resume: %d already vetted, %d left to vet", before - len(remaining), len(remaining))

    logger.info(
        "Vetting %d candidates from %s at speed=%.2f m/s (%s)",
        len(remaining),
        sources,
        args.speed,
        "in-process" if args.no_isolation else "crash-isolated workers",
    )

    try:
        if args.no_isolation:
            results = run_vetting(
                remaining, speed=args.speed, seed=args.seed, robot_uids=args.robot, on_result=on_result
            )
        else:
            isolation_kwargs = {} if args.timeout is None else {"object_timeout": args.timeout}
            results = run_isolated(
                remaining,
                speed=args.speed,
                seed=args.seed,
                robot_uids=args.robot,
                gso_manifest_path=args.manifest,
                cache_dir=cache_dir,
                on_result=on_result,
                max_zero_progress=args.max_zero_progress,
                **isolation_kwargs,
            )
    except ZeroProgressAbort:
        if (
            owns_results_claim
            and args.shard is None
            and args.results is not None
            and discard_unused_results_claim(args.results)
        ):
            logger.warning("released the unused --results claim on %s; the corrected re-run can use it", args.results)
        raise

    recorded.update({r.id: r for r in results})
    manifest_results = [recorded[c.id] for c in candidates if c.id in recorded]
    manifest = build_manifest_dict(
        manifest_results,
        speed=args.speed,
        seed=args.seed,
        sources=sources,
        cache_dir=cache_dir,
        robot_uids=args.robot,
        object_timeout=None if args.no_isolation else (args.timeout or _WORKER_OBJECT_TIMEOUT),
    )
    write_manifest(out_path, manifest)
    results = manifest_results

    rejected = [r for r in results if r.verdict == "reject"]
    by_source: dict[str, list[int]] = {}
    for r in results:
        bucket = by_source.setdefault(r.source, [0, 0])
        bucket[0] += 1
        bucket[1] += 1 if r.verdict == "pass" else 0
    by_reason: dict[str, int] = {}
    for r in rejected:
        key = r.reason.split(";")[0].split("(")[0].strip() or "unknown"
        by_reason[key] = by_reason.get(key, 0) + 1

    print("\n==== object vetting summary ====")
    print(f"total:    {len(results)}")
    print(f"passed:   {len(results) - len(rejected)}")
    print(f"rejected: {len(rejected)}")
    for src, (total, passed) in by_source.items():
        print(f"  {src}: {passed}/{total} passed")
    print(f"manifest: {out_path}")
    if by_reason:
        print("\n-- rejections by reason --")
        for reason, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            print(f"  {count:4d}  {reason}")
        print("\n-- rejected ids --")
        for r in rejected:
            print(f"  {r.id}: {r.reason}")


if __name__ == "__main__":
    main()
