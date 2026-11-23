"""Merge a vetting run's verdicts into the published `data/vetted_objects.json`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict

from pydantic import ConfigDict, TypeAdapter, ValidationError, with_config

from sweep_bench.assets.catalog_quality import thickness_reason
from sweep_bench.assets.mesh_import import (
    GsoManifest,
    default_manifest_path,
    load_manifest,
    write_json_atomic,
)
from sweep_bench.vetting import (
    _WORKER_OBJECT_TIMEOUT,
    VetResult,
    _procedural_by_id,
    stale_verdict_reason,
    summarize,
    write_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VETTED_PATH = _REPO_ROOT / "data" / "vetted_objects.json"


_CONFIG_KEYS = ("vet_speed", "seed", "robot", "object_timeout")


@with_config(ConfigDict(extra="allow"))
class VettedManifest(TypedDict, total=False):
    schema_version: int
    vet_speed: float
    seed: int
    robot: str
    object_timeout: float | None
    harness: str
    sources: list[str]
    gso_cache_dir: str
    entries: list[dict[str, object]]
    summary: dict[str, object]


_MANIFEST_SCHEMA = TypeAdapter(VettedManifest)


def _load(path: Path) -> tuple[VettedManifest, list[VetResult]]:
    try:
        manifest = _MANIFEST_SCHEMA.validate_json(path.read_bytes())
    except ValidationError as exc:
        raise SystemExit(f"{path}: not a vetted manifest: {exc}") from exc
    return manifest, [VetResult.from_dict(e) for e in manifest["entries"]]


def _config(manifest: VettedManifest) -> dict[str, object]:
    config: dict[str, object] = {k: manifest.get(k) for k in _CONFIG_KEYS}
    if "object_timeout" not in manifest:
        config["object_timeout"] = _WORKER_OBJECT_TIMEOUT
    return config


def _check_harness(incoming: VettedManifest, published: VettedManifest) -> None:
    a, b = incoming.get("harness"), published.get("harness")
    if a is None or b is None:
        print(
            f"\n-- WARNING: unrecorded vetting harness (incoming={a!r}, published={b!r}) --\n"
            f"   These verdicts may have been decided by different thresholds, and nothing here can tell."
        )
        return
    if a != b:
        raise SystemExit(
            f"refusing to merge: the incoming run was vetted by harness {a!r} and the published manifest "
            f"by {b!r}. A verdict is a decision, and a decision made by different thresholds or logic is a "
            f"different verdict -- re-vet the incoming ids under the published harness, or re-vet the whole "
            f"catalog under the new one."
        )


def _check_config(incoming: VettedManifest, published: VettedManifest) -> None:
    _check_harness(incoming, published)
    a, b = _config(incoming), _config(published)
    differing = {k: (a[k], b[k]) for k in _CONFIG_KEYS if a[k] != b[k]}
    if differing:
        raise SystemExit(
            "refusing to merge: the incoming run and the published manifest disagree on the vet configuration "
            f"{differing!r}. Verdicts are configuration-dependent and the merged file keeps one header."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", type=Path, required=True, help="Vetted manifest of the new run.")
    parser.add_argument("--vetted", type=Path, default=_VETTED_PATH, help="Published manifest to merge into.")
    parser.add_argument("--apply", action="store_true", help="Write the merge (default: report only).")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Imported-mesh manifest whose `pending_revet` debt these verdicts discharge (default: <cache>).",
    )
    args = parser.parse_args()

    incoming_manifest, incoming = _load(args.source)
    published, existing = _load(args.vetted)
    _check_config(incoming_manifest, published)

    manifest_path = args.manifest or default_manifest_path()
    entries_by_id = {e.id: e for e in load_manifest(manifest_path).entries} if manifest_path.is_file() else {}
    accepted: list[VetResult] = []
    stale: list[tuple[str, str]] = []
    for result in incoming:
        why = stale_verdict_reason(result, entries_by_id.get(result.id))
        if why is None:
            accepted.append(result)
        else:
            stale.append((result.id, why))
    if stale:
        print(f"\n-- refusing {len(stale)} incoming row(s) that did not measure the current object --")
        for object_id, why in stale:
            print(f"  {object_id}: {why}")
    incoming = accepted
    known = {r.id for r in existing}

    by_id = {r.id: r for r in incoming}
    duplicated = sorted({r.id for r in incoming if sum(1 for x in incoming if x.id == r.id) > 1})
    if duplicated:
        print(f"\n-- {len(duplicated)} incoming id(s) appeared more than once; keeping the LAST row of each --")
        for object_id in duplicated:
            print(f"  {object_id}")
    replaced = [r for r in by_id.values() if r.id in known]
    added = [r for r in by_id.values() if r.id not in known]
    merged = [by_id.get(r.id, r) for r in existing] + added

    passed_before = sum(1 for r in existing if r.verdict == "pass")
    passed_after = sum(1 for r in merged if r.verdict == "pass")
    print("\n==== merge vetted results ====")
    print(f"incoming:  {len(incoming)} ({len(added)} new ids, {len(replaced)} replacing existing)")
    print(f"candidates: {len(existing)} -> {len(merged)}")
    print(f"pass:       {passed_before} -> {passed_after}")
    for result in replaced:
        was = next(r for r in existing if r.id == result.id)
        if was.verdict != result.verdict:
            print(f"  {was.verdict:6s} -> {result.verdict:6s}  {result.id}   {result.reason}")
    rejected_new = [r for r in added if r.verdict == "reject"]
    if rejected_new:
        print(f"\n-- {len(rejected_new)} new ids rejected by the sim --")
        for result in rejected_new:
            print(f"  {result.id}: {result.reason}")

    if not args.apply:
        print("\n(report only; pass --apply to write)")
        return

    demoted: list[tuple[str, str]] = []
    for result in merged:
        if result.verdict != "pass":
            continue
        entry = entries_by_id.get(result.id)
        spec = _procedural_by_id().get(result.id)
        bbox = entry.bbox if entry is not None else (spec.bbox if spec is not None else None)
        if bbox is None:
            continue
        reason = thickness_reason(bbox, result.role)
        if reason:
            result.verdict = "reject"
            result.reason = reason
            demoted.append((result.id, reason))
    if demoted:
        print(f"\n-- {len(demoted)} row(s) demoted by the thickness screen before publication --")
        for object_id, reason in demoted:
            print(f"  {object_id}: {reason}")

    published["entries"] = [r.to_dict() for r in merged]
    published["summary"] = summarize(merged)
    published["sources"] = sorted({str(r.source) for r in merged})
    write_manifest(args.vetted, {str(k): v for k, v in published.items()})
    print(f"\nwrote {args.vetted}")

    if manifest_path.is_file():
        manifest = load_manifest(manifest_path)
        remaining = tuple(i for i in manifest.pending_revet if i not in by_id)
        if remaining != manifest.pending_revet:
            cleared = len(manifest.pending_revet) - len(remaining)
            updated = GsoManifest(
                schema_version=manifest.schema_version,
                source=manifest.source,
                license=manifest.license,
                cache_dir=manifest.cache_dir,
                entries=manifest.entries,
                rejections=manifest.rejections,
                pending_revet=remaining,
            )
            write_json_atomic(manifest_path, updated.to_dict())
            print(f"cleared {cleared} ids from pending_revet in {manifest_path} ({len(remaining)} left)")


if __name__ == "__main__":
    main()
