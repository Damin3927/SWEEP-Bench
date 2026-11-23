"""Move imported entries to the catalog role their geometry actually fits."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from sweep_bench.assets.mesh_import import (
    GsoManifest,
    default_cache_dir,
    default_manifest_path,
    load_manifest,
    write_json_atomic,
)

logger = logging.getLogger("reclassify_roles")

_TARGET_ROLES: dict[str, str] = {
    "ycb_030_fork": "tool",
    "ycb_032_knife": "tool",
    "ycb_037_scissors": "tool",
    "gso_diamond_visions_scissors_red": "tool",
    "ycb_042_adjustable_wrench": "tool",
    "ycb_048_hammer": "tool",
    "ycb_059_chain": "tool",
    "ycb_029_plate": "container",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="Write the manifest (default: report only).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cache_dir = args.cache_dir or default_cache_dir()
    manifest_path = args.manifest or default_manifest_path(cache_dir)
    manifest = load_manifest(manifest_path)
    by_id = {e.id: e for e in manifest.entries}

    missing = sorted(set(_TARGET_ROLES) - set(by_id))
    if missing:
        raise SystemExit(f"ids not in the imported-mesh manifest: {missing}")

    changed = {}
    print("\n==== role reclassification ====")
    for object_id, role in sorted(_TARGET_ROLES.items()):
        entry = by_id[object_id]
        already = entry.role == role and entry.patch_radius is not None and entry.grasp_width is not None
        if already:
            print(f"  {object_id:36s} already {role}")
            continue
        replacement = dataclasses.replace(
            entry,
            role=role,
            patch_radius=entry.patch_radius if entry.patch_radius is not None else min(entry.bbox[0], entry.bbox[1]),
            grasp_width=entry.grasp_width if entry.grasp_width is not None else 2.0 * min(entry.bbox[0], entry.bbox[1]),
            tags=(*(t for t in entry.tags if t != entry.role), role, f"reclassified_from:{entry.role}")
            if entry.role != role
            else entry.tags,
        )
        changed[object_id] = replacement
        thickness = 2000.0 * min(entry.bbox)
        was = entry.role if entry.role != role else f"{role} (restoring rider fields)"
        print(f"  {object_id:36s} {was} -> {role:16s} (smallest full dim {thickness:.2f} mm)")

    print(f"\nto reclassify: {len(changed)}")
    if not args.apply:
        print("\n(report only; pass --apply to write)")
        return
    if not changed:
        print("nothing to do")
        return

    pending = sorted(set(manifest.pending_revet) | set(changed))
    updated = GsoManifest(
        schema_version=manifest.schema_version,
        source=manifest.source,
        license=manifest.license,
        cache_dir=manifest.cache_dir,
        entries=tuple(changed.get(e.id, e) for e in manifest.entries),
        rejections=manifest.rejections,
        pending_revet=tuple(pending),
    )
    write_json_atomic(manifest_path, updated.to_dict())
    print(f"\nwrote {manifest_path}")
    manifest_moved = args.manifest is not None or args.cache_dir is not None
    passthrough = f" --manifest {manifest_path}" if manifest_moved else ""
    if args.cache_dir is not None:
        passthrough += f" --cache-dir {args.cache_dir}"
    print(
        "\nA role change moves an object to different sim gates, so its verdict is stale.\n"
        f"Recorded {len(pending)} ids as pending_revet. Measure and publish them with:\n"
        "  uv run --no-sync python scripts/vet_objects.py --sources all --only "
        + " ".join(sorted(changed))
        + f"{passthrough} --out /tmp/roles.json\n"
        "  uv run --no-sync python scripts/merge_vetted_results.py --from /tmp/roles.json"
        + (f" --manifest {manifest_path}" if manifest_moved else "")
        + " --apply"
    )


if __name__ == "__main__":
    main()
