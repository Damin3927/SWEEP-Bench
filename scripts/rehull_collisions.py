"""Replace a rider's multi-part collision geometry with a single convex hull."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from sweep_bench.assets.mesh_import import (
    default_cache_dir,
    default_manifest_path,
    load_manifest,
    load_mesh,
    write_bytes_atomic,
    write_json_atomic,
)


def hull_path(cache_dir: Path, object_id: str) -> Path:
    return cache_dir / "processed" / "hulls" / f"{object_id}.ply"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids-file", type=Path, required=True, help="One id per line; the screen decided these.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="Write the hulls and the manifest (default: dry run).")
    args = parser.parse_args()

    manifest_path: Path = args.manifest or default_manifest_path()
    cache_dir: Path = args.cache_dir or default_cache_dir()
    ids = [x.strip() for x in args.ids_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not ids:
        raise SystemExit(f"{args.ids_file}: no ids")

    manifest = load_manifest(manifest_path)
    by_id = {e.id: e for e in manifest.entries}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise SystemExit(f"{len(unknown)} id(s) are not in {manifest_path} (e.g. {unknown[:3]})")

    already = [i for i in ids if by_id[i].collision_method == "convex_hull"]
    if already:
        print(f"note: {len(already)} of {len(ids)} are already single hulls; they are left untouched")
    remaining = [i for i in ids if i not in set(already)]

    rewritten: dict[str, dict[str, object]] = {}
    for n, object_id in enumerate(remaining, 1):
        entry = by_id[object_id]
        mesh = load_mesh(Path(entry.collision_file))
        hull = mesh.convex_hull
        if not hull.is_watertight or hull.volume <= 0:
            raise SystemExit(
                f"{object_id}: convex hull is degenerate (watertight={hull.is_watertight}, vol={hull.volume})"
            )
        out = hull_path(cache_dir, object_id)
        rewritten[object_id] = {
            "from": entry.collision_file,
            "parts": entry.collision_parts,
            "to": str(out),
            "hull_verts": len(hull.vertices),
        }
        if args.apply:
            out.parent.mkdir(parents=True, exist_ok=True)
            write_bytes_atomic(out, hull.export(file_type="ply"))
        if n % 25 == 0:
            print(f"  {n}/{len(remaining)}", flush=True)

    print(f"\n{len(rewritten)} entries would be rewritten in {manifest_path}")
    for object_id, info in list(rewritten.items())[:3]:
        print(f"  {object_id}: {info['parts']} parts -> 1 hull ({info['hull_verts']} verts)")
    if not args.apply:
        print("dry run; re-run with --apply to write")
        return

    entries = tuple(
        dataclasses.replace(
            e,
            collision_file=str(hull_path(cache_dir, e.id)),
            collision_method="convex_hull",
            collision_parts=1,
        )
        if e.id in rewritten
        else e
        for e in manifest.entries
    )
    pending = tuple(sorted(set(manifest.pending_revet) | set(rewritten)))
    write_json_atomic(manifest_path, dataclasses.replace(manifest, entries=entries, pending_revet=pending).to_dict())
    print(f"wrote {manifest_path}; {len(pending)} id(s) now pending_revet")
    print(
        "\nTheir vetting verdicts were measured on the OLD shape. Discharge the debt:\n"
        f"  uv run --no-sync python scripts/vet_objects.py --sources robocasa \\\n"
        f"      --only $(tr '\\n' ' ' < {args.ids_file}) --out /tmp/rehulled.json\n"
        "  uv run --no-sync python scripts/merge_vetted_results.py --from /tmp/rehulled.json --apply\n"
        "then re-screen those ids -- a screening verdict is a measurement of the shape too."
    )


if __name__ == "__main__":
    main()
