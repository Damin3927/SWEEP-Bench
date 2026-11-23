"""Split merged multi-object scans into one importable object per real object."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sweep_bench.assets.catalog_quality import MERGED_SET_REASON
from sweep_bench.assets.mesh_import import (
    _MIN_BODY_VOLUME_FRAC,
    GSO_ATTRIBUTION_COLLECTION,
    GSO_SOURCE,
    POLYHAVEN_ATTRIBUTION_COLLECTION,
    POLYHAVEN_SOURCE,
    ROBOCASA_ATTRIBUTION_COLLECTION,
    ROBOCASA_SOURCE,
    YCB_ATTRIBUTION_COLLECTION,
    YCB_SOURCE,
    GsoManifest,
    ImportedMeshEntry,
    MeshRejected,
    body_face_groups,
    decompose_collision,
    default_cache_dir,
    default_manifest_path,
    ensure_visual_textures,
    load_manifest,
    load_mesh,
    mass_from_density,
    measure_mesh,
    write_json_atomic,
)
from sweep_bench.vetting import _WORKER_OBJECT_TIMEOUT, VetResult, summarize, write_manifest

if TYPE_CHECKING:
    import numpy as np
    import trimesh

logger = logging.getLogger("split_merged_sets")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VETTED_PATH = _REPO_ROOT / "data" / "vetted_objects.json"

_UNSPLITTABLE: tuple[str, ...] = ("gso_assorted_vegetable_set",)


_ATTRIBUTION_COLLECTIONS: dict[str, str] = {
    GSO_SOURCE: GSO_ATTRIBUTION_COLLECTION,
    YCB_SOURCE: YCB_ATTRIBUTION_COLLECTION,
    ROBOCASA_SOURCE: ROBOCASA_ATTRIBUTION_COLLECTION,
    POLYHAVEN_SOURCE: POLYHAVEN_ATTRIBUTION_COLLECTION,
}


def _collection_for(source: str) -> str:
    try:
        return _ATTRIBUTION_COLLECTIONS[source]
    except KeyError:
        raise ValueError(
            f"no attribution collection recorded for source {source!r}; crediting another "
            f"collection would be a licence bug — add the mapping before splitting its sets"
        ) from None


def merged_set_parents(vetted_path: Path) -> list[str]:
    with vetted_path.open(encoding="utf-8") as f:
        vetted = json.load(f)
    return [
        str(e["id"])
        for e in vetted["entries"]
        if str(e.get("reason", "")).startswith(MERGED_SET_REASON) and str(e["id"]) not in _UNSPLITTABLE
    ]


def _submesh(mesh: trimesh.Trimesh, faces: np.ndarray) -> trimesh.Trimesh:
    import trimesh as tm

    part = mesh.submesh([faces], append=True, repair=False)
    if not isinstance(part, tm.Trimesh):
        raise MeshRejected("submesh did not return a single mesh")
    return part


def _export_part(mesh: trimesh.Trimesh, faces: np.ndarray, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    part = _submesh(mesh, faces)
    obj_path = out_dir / "model.obj"
    part.export(str(obj_path))
    return obj_path


def split_parent(
    entry: ImportedMeshEntry, cache_dir: Path, *, density: float | None = None, write: bool = True
) -> tuple[list[ImportedMeshEntry], list[tuple[str, str]]]:
    density = entry.density if density is None else density
    mesh = load_mesh(Path(entry.visual_file))
    merged, groups = body_face_groups(mesh)
    if len(groups) < 2:
        return [], [(entry.id, f"split produced {len(groups)} body; nothing to separate")]

    import numpy as np

    faces = np.asarray(merged.faces, dtype=int)
    vertices = np.asarray(merged.vertices, dtype=float)
    volumes = []
    for selected in groups:
        from sweep_bench.assets.mesh_import import _points_hull_volume

        volumes.append(_points_hull_volume(vertices[np.unique(faces[selected])]))
    largest = max(volumes)

    parts: list[ImportedMeshEntry] = []
    rejected: list[tuple[str, str]] = []
    for index, (selected, volume) in enumerate(zip(groups, volumes, strict=True)):
        part_id = f"{entry.id}_part{index}"
        if volume < _MIN_BODY_VOLUME_FRAC * largest:
            continue
        part_dir = (cache_dir / "processed" / entry.name / "parts" / f"part{index}").resolve()
        obj_path = part_dir / "model.obj"
        collision_path = part_dir / "collision.ply"
        try:
            if write:
                obj_path = _export_part(merged, selected, part_dir)
                ensure_visual_textures(obj_path)
                part_mesh = load_mesh(obj_path)
            else:
                part_mesh = _submesh(merged, selected)
            bounds = part_mesh.bounds
            lo = (float(bounds[0][0]), float(bounds[0][1]), float(bounds[0][2]))
            hi = (float(bounds[1][0]), float(bounds[1][1]), float(bounds[1][2]))
            measurement = measure_mesh(lo, hi, model_id=part_id)
            if write:
                result = decompose_collision(part_mesh, collision_path, method="convex_hull")
                method, parts_count, native_volume = result.method, result.parts, result.native_volume
            else:
                hull = part_mesh.convex_hull
                method, parts_count, native_volume = "convex_hull", 1, float(abs(hull.volume))
        except MeshRejected as e:
            rejected.append((part_id, str(e)))
            continue
        parts.append(
            ImportedMeshEntry(
                id=part_id,
                name=f"{entry.name}__part{index}",
                role="graspable_rider",
                source=entry.source,
                license=entry.license,
                attribution=f"{entry.name}__part{index} (body {index} of {entry.name}, "
                f"{_collection_for(entry.source)}), {entry.license}, {entry.source_url}",
                source_url=entry.source_url,
                visual_file=str(obj_path),
                collision_file=str(collision_path),
                collision_method=method,
                collision_parts=parts_count,
                scale=measurement.scale,
                native_center=measurement.native_center,
                native_half_extents=measurement.native_half_extents,
                bbox=measurement.bbox,
                mass=mass_from_density(native_volume, measurement.scale, density),
                density=density,
                friction=entry.friction,
                patch_radius=measurement.patch_radius,
                grasp_width=measurement.grasp_width,
                tags=(*entry.tags, "split_part", f"parent:{entry.id}", f"body:{index}"),
            )
        )
    return parts, rejected


def _demote_obsolete(vetted_path: Path, obsolete: set[str]) -> None:
    if not obsolete:
        return
    with vetted_path.open(encoding="utf-8") as f:
        vetted = json.load(f)
    entries = [VetResult.from_dict(e) for e in vetted["entries"]]
    touched = 0
    for result in entries:
        if result.id in obsolete and result.verdict != "reject":
            result.verdict = "reject"
            result.reason = "split_obsolete (a forced re-split no longer produces this body)"
            touched += 1
    if not touched:
        return
    vetted["entries"] = [r.to_dict() for r in entries]
    vetted["summary"] = summarize(entries)
    write_manifest(vetted_path, vetted)
    print(f"demoted {touched} obsolete part row(s) in {vetted_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--vetted", type=Path, default=_VETTED_PATH)
    parser.add_argument("--report", action="store_true", help="Print each parent's bodies without writing.")
    parser.add_argument("--apply", action="store_true", help="Write the part meshes and manifest entries.")
    parser.add_argument("--force", action="store_true", help="Re-split parents whose parts already exist.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cache_dir = args.cache_dir or default_cache_dir()
    manifest_path = args.manifest or default_manifest_path(cache_dir)
    manifest = load_manifest(manifest_path)
    by_id = {e.id: e for e in manifest.entries}
    existing = set(by_id)

    parents = merged_set_parents(args.vetted)
    print("\n==== merged-set split ====")
    print(f"parents to split: {len(parents)}  (skipped as unsplittable: {list(_UNSPLITTABLE)})")

    if args.force and args.apply:
        at_risk = {
            e.id for e in manifest.entries if "split_part" in e.tags and e.id.rsplit("_part", 1)[0] in set(parents)
        }
        if at_risk:
            write_json_atomic(
                manifest_path,
                dataclasses.replace(
                    manifest, pending_revet=tuple(sorted(set(manifest.pending_revet) | at_risk))
                ).to_dict(),
            )
            print(f"marked {len(at_risk)} existing part(s) pending_revet before overwriting their meshes")

    all_parts: list[ImportedMeshEntry] = []
    all_rejected: list[tuple[str, str]] = []
    resplit_parents: set[str] = set()
    for parent_id in parents:
        entry = by_id.get(parent_id)
        if entry is None:
            all_rejected.append((parent_id, "parent is not in the imported-mesh manifest"))
            continue
        if not args.force and f"{parent_id}_part0" in existing:
            print(f"  {parent_id}: already split (use --force to redo)")
            continue
        parts, rejected = split_parent(entry, cache_dir, write=args.apply)
        resplit_parents.add(parent_id)
        all_parts.extend(parts)
        all_rejected.extend(rejected)
        print(f"  {parent_id}: {len(parts)} parts, {len(rejected)} refused")
        for part in parts:
            print(
                f"      {part.id:52s} {2000 * part.bbox[0]:5.1f}x{2000 * part.bbox[1]:5.1f}x"
                f"{2000 * part.bbox[2]:5.1f}mm  scale {part.scale:5.2f}  {1000 * part.mass:6.1f}g"
            )
        for part_id, why in rejected:
            print(f"      {part_id:52s} REFUSED: {why}")

    print(f"\ntotal new parts: {len(all_parts)}   refused: {len(all_rejected)}")
    if not args.apply:
        print("\n(report only; pass --apply to write)")
        return

    replacements = {p.id: p for p in all_parts}
    respilt = set(resplit_parents)
    obsolete = {
        e.id
        for e in manifest.entries
        if "split_part" in e.tags and e.id.rsplit("_part", 1)[0] in respilt and e.id not in replacements
    }
    if obsolete:
        print(f"dropping {len(obsolete)} part entries the re-split no longer produces: {sorted(obsolete)}")
    existing_part_ids = {e.id for e in manifest.entries}
    rewritten = {p.id for p in all_parts if p.id in existing_part_ids}
    if rewritten:
        print(f"{len(rewritten)} rewritten part(s) marked pending_revet (their published verdicts are now stale)")
    kept = tuple(replacements.pop(e.id, e) for e in manifest.entries if e.id not in obsolete)
    updated = GsoManifest(
        schema_version=manifest.schema_version,
        source=manifest.source,
        license=manifest.license,
        cache_dir=manifest.cache_dir,
        entries=kept + tuple(p for p in all_parts if p.id in replacements),
        rejections=manifest.rejections,
        pending_revet=tuple(sorted((set(manifest.pending_revet) | rewritten) - obsolete)),
    )
    _demote_obsolete(args.vetted, obsolete)
    write_json_atomic(manifest_path, updated.to_dict())
    print(f"\nwrote {manifest_path} ({len(updated.entries)} entries)")
    if not all_parts:
        print("\nNo new parts this run (every parent was already split). Nothing to vet; use --force to re-split.")
        return
    part_ids = " ".join(p.id for p in all_parts)
    sources = sorted({p.source for p in all_parts})
    source_arg = sources[0] if len(sources) == 1 else "all"
    manifest_moved = args.manifest is not None or args.cache_dir is not None
    vet_paths = "".join(
        f" {flag} {value}"
        for flag, value, given in (
            ("--manifest", manifest_path, manifest_moved),
            ("--cache-dir", cache_dir, args.cache_dir is not None),
        )
        if given
    )
    published_paths = "".join(
        f" {flag} {value}"
        for flag, value, given in (
            ("--vetted", args.vetted, args.vetted != _VETTED_PATH),
            ("--manifest", manifest_path, manifest_moved),
        )
        if given
    )
    with args.vetted.open(encoding="utf-8") as f:
        target_timeout = json.load(f).get("object_timeout", _WORKER_OBJECT_TIMEOUT)
    if target_timeout is None:
        vet_paths += " --no-isolation"
    elif float(target_timeout) != _WORKER_OBJECT_TIMEOUT:
        vet_paths += f" --timeout {target_timeout}"
    print(
        "\nNext, in order (see scripts/vet_objects.py for the full regeneration sequence):\n"
        f"  uv run --no-sync python scripts/vet_objects.py --sources {source_arg} --only {part_ids}"
        f"{vet_paths} --out /tmp/parts.json\n"
        f"  uv run --no-sync python scripts/merge_vetted_results.py --from /tmp/parts.json{published_paths} --apply"
    )


if __name__ == "__main__":
    main()
