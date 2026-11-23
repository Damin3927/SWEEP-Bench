"""Import RoboCasa's object library into the shared imported-mesh manifest."""

from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from sweep_bench.assets.mesh_import import (
    ROBOCASA_ATTRIBUTION_COLLECTION,
    ROBOCASA_LICENSE,
    ROBOCASA_SOURCE,
    ROBOCASA_SOURCE_URL,
    GsoManifest,
    ImportedMeshEntry,
    MeshRejected,
    RejectedModel,
    default_cache_dir,
    default_manifest_path,
    load_manifest,
    load_mesh,
    mass_from_density,
    measure_collision_file,
    measure_mesh,
    slug_for,
    write_bytes_atomic,
    write_json_atomic,
)

logger = logging.getLogger("import_robocasa")


def robocasa_objects_dir() -> Path:
    try:
        from mani_skill import ASSET_DIR

        root = Path(ASSET_DIR)
    except Exception:  # noqa: BLE001 — the fallback is ManiSkill's documented default
        root = Path.home() / ".maniskill" / "data"
    return root / "scene_datasets" / "robocasa_dataset" / "assets" / "objects" / "objaverse"


def list_objects(objects_dir: Path) -> list[tuple[str, str, Path]]:
    if not objects_dir.is_dir():
        raise SystemExit(
            f"RoboCasa objects not found at {objects_dir}. Download them with "
            "`python -m mani_skill.utils.download_asset RoboCasa -y`."
        )
    found: list[tuple[str, str, Path]] = []
    for model in sorted(objects_dir.rglob("model.xml")):
        if "__MACOSX" in model.parts:
            continue
        found.append((model.parent.parent.name, model.parent.name, model))
    return found


def _meshes(model_xml: Path) -> tuple[list[tuple[Path, float]], list[tuple[Path, float]]]:
    root = ET.parse(model_xml).getroot()
    visual: list[tuple[Path, float]] = []
    collision: list[tuple[Path, float]] = []
    for mesh in root.iter("mesh"):
        file_attr = mesh.get("file")
        if not file_attr:
            continue
        scale_attr = (mesh.get("scale") or "1 1 1").split()
        scale = float(scale_attr[0])
        path = (model_xml.parent / file_attr).resolve()
        if not path.is_file():
            continue
        (collision if "coll" in (mesh.get("name") or "") else visual).append((path, scale))
    return visual, collision


def process_object(
    category: str, name: str, model_xml: Path, cache_dir: Path, density: float = 1000.0
) -> ImportedMeshEntry:
    import trimesh

    visual_files, collision_files = _meshes(model_xml)
    if not visual_files:
        raise MeshRejected("MJCF names no visual mesh")
    if not collision_files:
        raise MeshRejected("MJCF names no collision mesh")

    object_id = slug_for(f"{category}_{name}", prefix=ROBOCASA_SOURCE)
    processed = cache_dir / "processed" / ROBOCASA_SOURCE / category / name
    processed.mkdir(parents=True, exist_ok=True)

    visual_path, mjcf_scale = visual_files[0]

    collision_path = processed / "collision.ply"
    if not collision_path.is_file():
        merged = trimesh.util.concatenate([load_mesh(path) for path, _ in collision_files])
        write_bytes_atomic(collision_path, merged.export(file_type="ply"))
    result = measure_collision_file(collision_path)

    mesh = load_mesh(visual_path)
    bounds = mesh.bounds
    lo = (float(bounds[0][0]) * mjcf_scale, float(bounds[0][1]) * mjcf_scale, float(bounds[0][2]) * mjcf_scale)
    hi = (float(bounds[1][0]) * mjcf_scale, float(bounds[1][1]) * mjcf_scale, float(bounds[1][2]) * mjcf_scale)
    measurement = measure_mesh(lo, hi, model_id=object_id)
    total_scale = measurement.scale * mjcf_scale
    cx, cy, cz = measurement.native_center
    native_center = (cx / mjcf_scale, cy / mjcf_scale, cz / mjcf_scale)
    hx, hy, hz = measurement.native_half_extents
    native_half = (hx / mjcf_scale, hy / mjcf_scale, hz / mjcf_scale)

    return ImportedMeshEntry(
        id=object_id,
        name=f"{category}/{name}",
        role="graspable_rider",
        source=ROBOCASA_SOURCE,
        license=ROBOCASA_LICENSE,
        attribution=f"{category}/{name} ({ROBOCASA_ATTRIBUTION_COLLECTION}, Objaverse-sourced), "
        f"{ROBOCASA_LICENSE}, {ROBOCASA_SOURCE_URL}",
        source_url=ROBOCASA_SOURCE_URL,
        visual_file=str(visual_path),
        collision_file=str(collision_path),
        collision_method=result.method,
        collision_parts=result.parts,
        scale=total_scale,
        native_center=native_center,
        native_half_extents=native_half,
        bbox=measurement.bbox,
        mass=mass_from_density(result.native_volume, total_scale, density),
        density=density,
        friction=None,
        patch_radius=measurement.patch_radius,
        grasp_width=measurement.grasp_width,
        tags=(ROBOCASA_SOURCE, "mesh", "graspable_rider", "robocasa-objaverse", category),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--objects-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cache_dir = args.cache_dir or default_cache_dir()
    out_path = args.out or default_manifest_path(cache_dir)

    entries: list[ImportedMeshEntry] = []
    rejections: list[RejectedModel] = []
    pending_revet: tuple[str, ...] = ()
    source_header = ROBOCASA_SOURCE
    judged: set[str] = set()
    if args.merge and out_path.is_file():
        existing = load_manifest(out_path)
        entries.extend(existing.entries)
        rejections.extend(existing.rejections)
        pending_revet = existing.pending_revet
        source_header = ",".join(sorted({e.source for e in existing.entries} | {ROBOCASA_SOURCE}))
        judged = {e.id for e in existing.entries} | {r.id for r in rejections}

    def flush() -> None:
        write_json_atomic(
            out_path,
            GsoManifest(
                schema_version=1,
                source=source_header,
                license=ROBOCASA_LICENSE,
                cache_dir=str(cache_dir.resolve()),
                entries=tuple(entries),
                rejections=tuple(rejections),
                pending_revet=pending_revet,
            ).to_dict(),
        )

    objects = list_objects(args.objects_dir or robocasa_objects_dir())
    remaining = [o for o in objects if slug_for(f"{o[0]}_{o[1]}", ROBOCASA_SOURCE) not in judged]
    if args.limit is not None:
        remaining = remaining[: args.limit]
    logger.info(
        "RoboCasa: %d objects found, %d already judged, %d to process", len(objects), len(judged), len(remaining)
    )

    imported = refused = 0
    for position, (category, name, model_xml) in enumerate(remaining, start=1):
        try:
            entries.append(process_object(category, name, model_xml, cache_dir))
            imported += 1
            if position % 25 == 0 or imported <= 3:
                logger.info("[%d/%d] imported %s/%s", position, len(remaining), category, name)
        except MeshRejected as e:
            rejections.append(
                RejectedModel(
                    name=f"{category}/{name}", id=slug_for(f"{category}_{name}", ROBOCASA_SOURCE), reason=str(e)
                )
            )
            refused += 1
        except (OSError, ValueError, ET.ParseError) as e:
            logger.warning("[%d/%d] SKIP %s/%s (not judged): %s", position, len(remaining), category, name, e)
            continue
        if position % 25 == 0:
            flush()

    flush()
    print("\n==== RoboCasa import ====")
    print(f"found:    {len(objects)}")
    print(f"imported: {imported}")
    print(f"refused:  {refused}")
    print(f"manifest: {out_path} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
