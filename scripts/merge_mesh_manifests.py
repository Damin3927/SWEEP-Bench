"""Merge imported-mesh manifests into the one the runtime actually reads."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from sweep_bench.assets.mesh_import import (
    MANIFEST_SCHEMA_VERSION,
    GsoManifest,
    ImportedMeshEntry,
    RejectedModel,
    default_manifest_path,
    load_located_entries,
    load_manifest,
    write_json_atomic,
)
from sweep_bench.assets.vetted import committed_pass_ids, resolve_object_specs


def _conflicts(a: ImportedMeshEntry, b: ImportedMeshEntry) -> list[str]:
    da, db = a.to_dict(), b.to_dict()
    return sorted(k for k in da if da[k] != db[k])


def merge(manifests: list[GsoManifest]) -> GsoManifest:
    entries: dict[str, ImportedMeshEntry] = {}
    disagreements: list[str] = []
    for manifest in manifests:
        for entry in manifest.entries:
            seen = entries.get(entry.id)
            if seen is None:
                entries[entry.id] = entry
            elif seen != entry:
                disagreements.append(f"{entry.id}: {', '.join(_conflicts(seen, entry))}")
    if disagreements:
        shown = "\n  ".join(disagreements[:10])
        more = f"\n  ... and {len(disagreements) - 10} more" if len(disagreements) > 10 else ""
        raise SystemExit(
            f"refusing to merge: {len(disagreements)} id(s) carry different geometry in different "
            f"manifests, so no merge can be written without republishing some verdict under a shape "
            f"that did not produce it. Re-import or re-vet these instead:\n  {shown}{more}"
        )

    rejections: dict[tuple[str, str], RejectedModel] = {}
    for manifest in manifests:
        for rejection in manifest.rejections:
            rejections.setdefault((rejection.id, rejection.reason), rejection)

    pending = sorted({i for m in manifests for i in m.pending_revet})

    sources = ",".join(sorted({e.source for e in entries.values()}))
    licenses = ", ".join(sorted({e.license for e in entries.values()}))
    cache_dirs = ",".join(sorted({d for m in manifests for d in m.cache_dir.split(",") if d}))
    return GsoManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        source=sources,
        license=licenses,
        cache_dir=cache_dirs,
        entries=tuple(entries.values()),
        rejections=tuple(rejections.values()),
        pending_revet=tuple(pending),
    )


def _pending_ids(manifest_path: str | None) -> set[str]:
    path = Path(manifest_path) if manifest_path is not None else default_manifest_path()
    return set(load_manifest(path).pending_revet) if path.is_file() else set()


def _missing_mesh_ids(ids: set[str], manifest_path: str | None) -> set[str]:
    path = Path(manifest_path) if manifest_path is not None else default_manifest_path()
    if not path.is_file():
        return set()
    return {entry.id for entry in load_located_entries(path) if entry.id in ids and not entry.has_mesh_files()}


def report_coverage(label: str, manifest_path: str | None = None) -> int:
    published = committed_pass_ids()
    withheld = _pending_ids(manifest_path)
    published = [i for i in published if i not in withheld]
    resolved = resolve_object_specs(tuple(published), gso_manifest_path=manifest_path)
    gone = _missing_mesh_ids(set(published), manifest_path)
    buildable = [spec for spec in resolved if spec.id not in gone]
    shortfall = len(published) - len(buildable)
    print(f"{label}: {len(published)} committed pass ids -> {len(buildable)} buildable here ({shortfall} missing)")
    if withheld:
        print(
            f"  {len(withheld)} further id(s) are WITHHELD pending a re-vet, so they are outside this count "
            f"(e.g. {sorted(withheld)[:3]}). Publish their new verdicts to bring them back."
        )
    if shortfall:
        served = {spec.id for spec in buildable}
        by_prefix = Counter(i.split("_", 1)[0] for i in published if i not in served)
        print("  missing by id prefix: " + ", ".join(f"{p}_* {n}" for p, n in sorted(by_prefix.items())))
        if gone:
            print(
                f"  of those, {len(gone)} are in the manifest but their mesh files are gone (e.g. {sorted(gone)[:3]})"
            )
    return shortfall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--from",
        dest="sources",
        type=Path,
        nargs="+",
        default=[],
        help="Manifest(s) to merge IN, in precedence order (ties are refused, not resolved).",
    )
    parser.add_argument(
        "--into",
        type=Path,
        default=None,
        help=f"Manifest the runtime reads (default: {default_manifest_path()}).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Report published-vs-buildable on this machine and exit 1 on a shortfall; write nothing.",
    )
    parser.add_argument("--apply", action="store_true", help="Write the merged manifest (default: dry run).")
    args = parser.parse_args()

    into: Path = args.into or default_manifest_path()

    if args.check_only:
        if args.sources or args.apply:
            parser.error("--check-only takes no --from and does not write; drop --apply")
        raise SystemExit(1 if report_coverage("catalog coverage") else 0)

    if not args.sources:
        parser.error("--from is required (or use --check-only)")

    manifests: list[GsoManifest] = []
    if into.is_file():
        manifests.append(load_manifest(into))
    else:
        print(f"note: {into} does not exist yet; it will be created from --from alone")
    for path in args.sources:
        if not path.is_file():
            raise SystemExit(f"--from {path}: not a file")
        manifests.append(load_manifest(path))

    merged = merge(manifests)
    existing = len(manifests[0].entries) if into.is_file() else 0
    print(f"{into}: {existing} entries -> {len(merged.entries)} after merge (+{len(merged.entries) - existing})")
    print(f"  sources: {merged.source}")
    print(f"  rejections: {len(merged.rejections)}   pending_revet: {len(merged.pending_revet)}")

    if not args.apply:
        print("dry run; re-run with --apply to write")
        return

    write_json_atomic(into, merged.to_dict())
    print(f"wrote {into}")
    raise SystemExit(1 if report_coverage("catalog coverage after merge", str(into)) else 0)


if __name__ == "__main__":
    main()
