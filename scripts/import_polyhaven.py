"""Import Poly Haven models into the shared imported-mesh manifest."""

from __future__ import annotations

import argparse
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from sweep_bench.assets.mesh_import import (
    POLYHAVEN_API,
    POLYHAVEN_ATTRIBUTION_COLLECTION,
    POLYHAVEN_LICENSE,
    POLYHAVEN_SOURCE,
    POLYHAVEN_SOURCE_URL,
    GsoManifest,
    ImportedMeshEntry,
    MeshRejected,
    RejectedModel,
    decompose_collision,
    default_cache_dir,
    default_manifest_path,
    load_manifest,
    load_mesh,
    mass_from_density,
    measure_mesh,
    slug_for,
    within_root,
    write_bytes_atomic,
    write_json_atomic,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import trimesh

logger = logging.getLogger("import_polyhaven")

_MAX_NATIVE_EXTENT = 0.60
_MIN_NATIVE_EXTENT = 0.02

_RESOLUTION = "1k"


_HEADERS = {
    "User-Agent": "sweep-bench/1.0 research asset importer",
    "Accept": "*/*",
}


_ALLOWED_HOSTS = ("polyhaven.com", "polyhaven.org")


def _check_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise MeshRejected(f"refusing to fetch {url!r}: only https is allowed (got scheme {parsed.scheme!r})")
    host = (parsed.hostname or "").lower()
    if not any(host == allowed or host.endswith(f".{allowed}") for allowed in _ALLOWED_HOSTS):
        raise MeshRejected(f"refusing to fetch {url!r}: host {host!r} is not one of {_ALLOWED_HOSTS}")
    return url


class _CheckedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201 — urllib's signature
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_CheckedRedirectHandler())


def _open(url: str, timeout: float):
    request = urllib.request.Request(_check_url(url), headers=_HEADERS)  # noqa: S310 — scheme/host checked above
    # noqa rationale: the request URL is checked above and every redirect hop is
    return _OPENER.open(request, timeout=timeout)


def _json_object(value: object, what: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MeshRejected(f"{what} is not a JSON object")
    return {str(k): v for k, v in value.items()}


def api_get(url: str, timeout: float = 60.0) -> object:
    with _open(url, timeout) as response:
        return json.loads(response.read().decode())


def list_models() -> dict[str, dict]:
    payload = api_get(f"{POLYHAVEN_API}/assets?type=models")
    if not isinstance(payload, dict):
        raise ValueError("Poly Haven asset listing is not a JSON object")
    return {str(k): v for k, v in payload.items() if isinstance(v, dict)}


def download_asset(slug: str, raw_root: Path) -> Path:
    files = _json_object(api_get(f"{POLYHAVEN_API}/files/{slug}"), f"{slug}: file listing")
    gltf_section = _json_object(files.get("gltf"), f"{slug}: no glTF available")
    if not gltf_section:
        raise MeshRejected(f"{slug}: glTF section lists no resolutions")
    resolution = _RESOLUTION if _RESOLUTION in gltf_section else next(iter(gltf_section))
    entry = _json_object(
        _json_object(gltf_section[resolution], f"{slug}: glTF resolution block").get("gltf"),
        f"{slug}: glTF entry",
    )
    if "url" not in entry:
        raise MeshRejected(f"{slug}: glTF entry has no url")

    if slug != Path(slug).name or slug in ("", ".", ".."):
        raise MeshRejected(f"asset slug {slug!r} is not a single path component; refusing to build a cache path")
    out_dir = raw_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    gltf_path = out_dir / f"{slug}.gltf"
    _fetch(str(entry["url"]), gltf_path)
    include = entry.get("include")
    include_map = _json_object(include, f"{slug}: include block") if include else {}
    resolved_root = out_dir.resolve()
    for relative, spec in include_map.items():
        url = _json_object(spec, f"{slug}: include entry").get("url") if isinstance(spec, dict) else spec
        if url:
            target = out_dir / str(relative)
            if not within_root(resolved_root, target):
                raise MeshRejected(
                    f"{slug}: include entry {relative!r} resolves outside the asset directory {resolved_root}; "
                    f"refusing to download into it"
                )
            _fetch(str(url), target)
    return gltf_path


def _fetch(url: str, target: Path) -> None:
    if target.is_file() and target.stat().st_size > 0:
        return
    with _open(url, 180.0) as response:
        write_bytes_atomic(target, response.read())


def _merged_license(entries: Sequence[ImportedMeshEntry]) -> str:
    licenses = sorted({entry.license for entry in entries})
    if not licenses:
        return POLYHAVEN_LICENSE
    return ", ".join(licenses)


def _export_obj_with_materials(mesh: trimesh.Trimesh, visual_path: Path) -> None:
    from trimesh.exchange.obj import export_obj

    obj_text, sidecars = export_obj(mesh, include_texture=True, return_texture=True, mtl_name="model.mtl")
    for name, payload in (sidecars or {}).items():
        target = visual_path.parent / name
        if not within_root(visual_path.parent.resolve(), target):
            raise MeshRejected(f"{visual_path.name}: export sidecar {name!r} resolves outside {visual_path.parent}")
        write_bytes_atomic(target, payload if isinstance(payload, bytes) else str(payload).encode())
    write_bytes_atomic(visual_path, obj_text.encode())


def check_native_size(mesh: trimesh.Trimesh, slug: str) -> None:
    extents = mesh.bounds[1] - mesh.bounds[0]
    largest = float(max(extents))
    if largest > _MAX_NATIVE_EXTENT:
        raise MeshRejected(
            f"native size {largest:.2f} m exceeds the handheld ceiling {_MAX_NATIVE_EXTENT} m "
            "(furniture/terrain scale; normalizing it would rename a sofa an 8 cm object)"
        )
    if largest < _MIN_NATIVE_EXTENT:
        raise MeshRejected(f"native size {largest:.3f} m below the {_MIN_NATIVE_EXTENT} m floor")


def process_asset(slug: str, meta: dict, cache_dir: Path, density: float = 1000.0) -> ImportedMeshEntry:
    gltf_path = download_asset(slug, cache_dir / "raw" / POLYHAVEN_SOURCE)
    mesh = load_mesh(gltf_path)
    check_native_size(mesh, slug)

    object_id = slug_for(slug, prefix=POLYHAVEN_SOURCE)
    processed = cache_dir / "processed" / POLYHAVEN_SOURCE / slug
    processed.mkdir(parents=True, exist_ok=True)
    visual_path = processed / "model.obj"
    if not visual_path.is_file():
        _export_obj_with_materials(mesh, visual_path)
    exported = load_mesh(visual_path)

    bounds = exported.bounds
    lo = (float(bounds[0][0]), float(bounds[0][1]), float(bounds[0][2]))
    hi = (float(bounds[1][0]), float(bounds[1][1]), float(bounds[1][2]))
    measurement = measure_mesh(lo, hi, model_id=object_id)
    collision_path = processed / "collision.ply"
    result = decompose_collision(exported, collision_path, method="convex_hull")

    authors = ", ".join((meta.get("authors") or {}).keys()) or "Poly Haven"
    return ImportedMeshEntry(
        id=object_id,
        name=str(meta.get("name") or slug),
        role="graspable_rider",
        source=POLYHAVEN_SOURCE,
        license=POLYHAVEN_LICENSE,
        attribution=f"{meta.get('name') or slug} by {authors} ({POLYHAVEN_ATTRIBUTION_COLLECTION}), "
        f"{POLYHAVEN_LICENSE}, {POLYHAVEN_SOURCE_URL}/a/{slug}",
        source_url=f"{POLYHAVEN_SOURCE_URL}/a/{slug}",
        visual_file=str(visual_path),
        collision_file=str(collision_path),
        collision_method=result.method,
        collision_parts=result.parts,
        scale=measurement.scale,
        native_center=measurement.native_center,
        native_half_extents=measurement.native_half_extents,
        bbox=measurement.bbox,
        mass=mass_from_density(result.native_volume, measurement.scale, density),
        density=density,
        friction=None,
        patch_radius=measurement.patch_radius,
        grasp_width=measurement.grasp_width,
        tags=(POLYHAVEN_SOURCE, "mesh", "graspable_rider", *(str(c) for c in (meta.get("categories") or [])[:4])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="Manifest path (default: <cache>/gso_manifest.json).")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N assets (a sample run).")
    parser.add_argument("--merge", action="store_true", help="Merge into an existing manifest, skipping judged ids.")
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
    source_header = POLYHAVEN_SOURCE
    judged: set[str] = set()
    if args.merge and out_path.is_file():
        existing = load_manifest(out_path)
        entries.extend(existing.entries)
        rejections.extend(existing.rejections)
        pending_revet = existing.pending_revet
        source_header = ",".join(sorted({e.source for e in existing.entries} | {POLYHAVEN_SOURCE}))
        judged = {e.id for e in existing.entries} | {r.id for r in rejections}

    def flush() -> None:
        manifest = GsoManifest(
            schema_version=1,
            source=source_header,
            license=_merged_license(entries),
            cache_dir=str(cache_dir.resolve()),
            entries=tuple(entries),
            rejections=tuple(rejections),
            pending_revet=pending_revet,
        )
        write_json_atomic(out_path, manifest.to_dict())

    models = list_models()
    remaining = [
        (slug, meta) for slug, meta in sorted(models.items()) if slug_for(slug, POLYHAVEN_SOURCE) not in judged
    ]
    if args.limit is not None:
        remaining = remaining[: args.limit]
    logger.info(
        "Poly Haven: %d models listed, %d already judged, %d to process", len(models), len(judged), len(remaining)
    )

    imported = refused = 0
    for position, (slug, meta) in enumerate(remaining, start=1):
        try:
            entries.append(process_asset(slug, meta, cache_dir))
            imported += 1
            logger.info("[%d/%d] imported %s", position, len(remaining), slug)
        except MeshRejected as e:
            rejections.append(RejectedModel(name=slug, id=slug_for(slug, POLYHAVEN_SOURCE), reason=str(e)))
            refused += 1
            logger.info("[%d/%d] REJECT %s: %s", position, len(remaining), slug, e)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            logger.warning("[%d/%d] SKIP %s (not judged): %s", position, len(remaining), slug, e)
            continue
        flush()

    flush()
    print("\n==== Poly Haven import ====")
    print(f"listed:   {len(models)}")
    print(f"imported: {imported}")
    print(f"refused:  {refused}")
    print(f"manifest: {out_path} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
