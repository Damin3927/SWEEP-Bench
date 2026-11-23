"""Import scanned-object collections into a candidate object manifest."""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING

from sweep_bench.assets.mesh_import import (
    CC_BY_4_LICENSE_NAME,
    GSO_ATTRIBUTION_COLLECTION,
    GSO_LICENSE,
    GSO_SOURCE,
    YCB_ATTRIBUTION_COLLECTION,
    YCB_LICENSE,
    YCB_SOURCE,
    YCB_SOURCE_URL,
    CollisionMethod,
    DecompositionResult,
    DecompositionUnavailable,
    GsoManifest,
    ImportedMeshEntry,
    MeshRejected,
    RejectedModel,
    coacd_available,
    decompose_collision,
    default_cache_dir,
    default_manifest_path,
    ensure_visual_textures,
    load_mesh,
    mass_from_density,
    measure_collision_file,
    measure_mesh,
    slug_for,
    write_json_atomic,
    ycb_models_dir,
)

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger("import_objects")

_FUEL_BASE = "https://fuel.gazebosim.org/1.0/GoogleResearch"
_USER_AGENT = "sweep-bench-gso-importer/1.0"
_CC_BY_TOKEN = "attribution 4.0"


def _http_get(url: str, *, timeout: float = 60.0, retries: int = 3) -> tuple[bytes, dict[str, str]]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), {k.lower(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429:
                raise
            last = e
            wait = 2.0 * (attempt + 1)
            logger.warning("GET %s failed (attempt %d/%d): %s; retrying in %.0fs", url, attempt + 1, retries, e, wait)
            sleep(wait)
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            wait = 2.0 * (attempt + 1)
            logger.warning("GET %s failed (attempt %d/%d): %s; retrying in %.0fs", url, attempt + 1, retries, e, wait)
            sleep(wait)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def _http_json(url: str) -> object:
    body, _ = _http_get(url)
    return json.loads(body)


def _field(value: object, key: str) -> object | None:
    if isinstance(value, dict):
        for k, v in value.items():
            if k == key:
                return v
    return None


def model_page_url(name: str) -> str:
    return f"{_FUEL_BASE}/models/{urllib.parse.quote(name)}"


def _total_count(headers: dict[str, str]) -> int | None:
    raw = headers.get("x-total-count")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("ignoring unparsable X-Total-Count header %r", raw)
        return None


def list_models(limit: int, offset: int, page_size: int = 100) -> list[dict[str, object]]:
    models: list[dict[str, object]] = []
    page = 1 + offset // page_size
    skip = offset % page_size
    total: int | None = None
    while len(models) < limit:
        if total is not None and (page - 1) * page_size >= total:
            break
        url = f"{_FUEL_BASE}/models?per_page={page_size}&page={page}"
        try:
            body, headers = _http_get(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.debug("page %d is past the end of the listing (404); stopping", page)
                break
            raise
        batch = json.loads(body)
        if not isinstance(batch, list) or not batch:
            break
        if total is None:
            total = _total_count(headers)
            logger.info("Fuel reports %s models in the listing", total if total is not None else "an unknown number")
        for item in batch:
            if skip > 0:
                skip -= 1
                continue
            if isinstance(item, dict):
                models.append({str(k): v for k, v in item.items()})
            if len(models) >= limit:
                break
        if len(batch) < page_size:
            break
        page += 1
    return models


def is_cc_by_4(model: dict[str, object]) -> bool:
    name = str(model.get("license_name", ""))
    return _CC_BY_TOKEN in name.lower()


def resolve_version(model: dict[str, object]) -> int:
    version = model.get("version")
    if isinstance(version, int) and version > 0:
        return version
    detail_version = _field(_http_json(model_page_url(str(model["name"]))), "version")
    if isinstance(detail_version, int) and detail_version > 0:
        return detail_version
    return 1


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    root = dest.resolve()
    for member in zf.infolist():
        target = (root / member.filename).resolve()
        if root != target and root not in target.parents:
            raise MeshRejected(f"unsafe zip member path: {member.filename!r}")
    zf.extractall(root)


def _safe_model_dirname(name: str) -> str:
    if not name or name in (".", "..") or name != Path(name).name:
        raise ValueError(
            f"Fuel model name {name!r} is not a single path component, so it cannot name a cache "
            f"directory. Refusing to derive a cache path from it."
        )
    return name


def download_model(name: str, version: int, raw_root: Path, *, refetch: bool = False) -> Path:
    name = _safe_model_dirname(name)
    dest = raw_root / name
    if refetch:
        shutil.rmtree(dest, ignore_errors=True)
    else:
        obj = find_obj(dest)
        if obj is not None:
            return obj
    url = f"{_FUEL_BASE}/models/{urllib.parse.quote(name)}/{version}/{urllib.parse.quote(name)}.zip"
    body, _ = _http_get(url, timeout=180.0)
    staging = raw_root / f".staging-{name}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            _safe_extract(zf, staging)
        if find_obj(staging) is None:
            raise MeshRejected("no meshes/model.obj in the downloaded archive")
        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    obj = find_obj(dest)
    if obj is None:
        raise MeshRejected("no meshes/model.obj in the downloaded archive")
    return obj


def find_obj(model_dir: Path) -> Path | None:
    preferred = model_dir / "meshes" / "model.obj"
    if preferred.is_file():
        return preferred
    for path in sorted(model_dir.rglob("*.obj")):
        return path
    return None


def process_model(
    model: dict[str, object],
    *,
    cache_dir: Path,
    method: CollisionMethod,
    density: float,
    friction: float | None,
    allow_hull_fallback: bool,
) -> ImportedMeshEntry:
    name = str(model["name"])
    version = resolve_version(model)
    raw_root = cache_dir / "raw"
    obj_path = download_model(name, version, raw_root)
    try:
        ensure_visual_textures(obj_path)
    except MeshRejected:
        logger.warning("%s: texture check failed on the cached extract, re-downloading once", name)
        obj_path = download_model(name, version, raw_root, refetch=True)
        ensure_visual_textures(obj_path)

    mesh = load_mesh(obj_path)
    bounds = mesh.bounds
    lo = (float(bounds[0][0]), float(bounds[0][1]), float(bounds[0][2]))
    hi = (float(bounds[1][0]), float(bounds[1][1]), float(bounds[1][2]))
    measurement = measure_mesh(lo, hi, model_id=slug_for(name))

    collision_path = cache_dir / "processed" / name / "collision.ply"
    used_method, result = _decompose(mesh, collision_path, method=method, allow_hull_fallback=allow_hull_fallback)
    mass = mass_from_density(result.native_volume, measurement.scale, density)

    source_url = model_page_url(name)
    attribution = f"{name} ({GSO_ATTRIBUTION_COLLECTION}), {GSO_LICENSE}, {source_url}"
    return ImportedMeshEntry(
        id=slug_for(name),
        name=name,
        role="graspable_rider",
        source=GSO_SOURCE,
        license=GSO_LICENSE,
        attribution=attribution,
        source_url=source_url,
        visual_file=str(obj_path.resolve()),
        collision_file=str(collision_path.resolve()),
        collision_method=used_method,
        collision_parts=result.parts,
        scale=measurement.scale,
        native_center=measurement.native_center,
        native_half_extents=measurement.native_half_extents,
        bbox=measurement.bbox,
        mass=mass,
        density=density,
        friction=friction,
        patch_radius=measurement.patch_radius,
        grasp_width=measurement.grasp_width,
        tags=("gso", "mesh", "graspable_rider"),
    )


def list_ycb_models(models_dir: Path) -> list[dict[str, object]]:
    if not models_dir.is_dir():
        raise RuntimeError(
            f"YCB models not found at {models_dir}. ManiSkill downloads them with "
            "`python -m mani_skill.utils.download_asset ycb`."
        )
    return [
        {"name": model_dir.name, "license_name": CC_BY_4_LICENSE_NAME}
        for model_dir in sorted(models_dir.iterdir())
        if model_dir.is_dir()
    ]


def process_ycb_model(
    model: dict[str, object],
    *,
    models_dir: Path,
    density: float,
    friction: float | None,
) -> ImportedMeshEntry:
    name = str(model["name"])
    model_dir = models_dir / name
    visual = model_dir / "textured.obj"
    collision = model_dir / "collision.ply"
    for label, path in (("visual", visual), ("collision", collision)):
        if not path.is_file():
            raise MeshRejected(f"YCB model {name} has no {label} mesh at {path}")

    ensure_visual_textures(visual)

    mesh = load_mesh(visual)
    bounds = mesh.bounds
    lo = (float(bounds[0][0]), float(bounds[0][1]), float(bounds[0][2]))
    hi = (float(bounds[1][0]), float(bounds[1][1]), float(bounds[1][2]))
    measurement = measure_mesh(lo, hi, model_id=slug_for(name, prefix=YCB_SOURCE))

    result = measure_collision_file(collision)
    mass = mass_from_density(result.native_volume, measurement.scale, density)

    attribution = f"{name} ({YCB_ATTRIBUTION_COLLECTION}), {YCB_LICENSE}, {YCB_SOURCE_URL}"
    return ImportedMeshEntry(
        id=slug_for(name, prefix=YCB_SOURCE),
        name=name,
        role="graspable_rider",
        source=YCB_SOURCE,
        license=YCB_LICENSE,
        attribution=attribution,
        source_url=YCB_SOURCE_URL,
        visual_file=str(visual.resolve()),
        collision_file=str(collision.resolve()),
        collision_method=result.method,
        collision_parts=result.parts,
        scale=measurement.scale,
        native_center=measurement.native_center,
        native_half_extents=measurement.native_half_extents,
        bbox=measurement.bbox,
        mass=mass,
        density=density,
        friction=friction,
        patch_radius=measurement.patch_radius,
        grasp_width=measurement.grasp_width,
        tags=("ycb", "mesh", "graspable_rider"),
    )


def _decompose(
    mesh: trimesh.Trimesh, out_path: Path, *, method: CollisionMethod, allow_hull_fallback: bool
) -> tuple[CollisionMethod, DecompositionResult]:
    try:
        return method, decompose_collision(mesh, out_path, method=method)
    except MeshRejected:
        if method == "coacd" and allow_hull_fallback:
            logger.warning("CoACD failed for %s; falling back to a convex hull", out_path.parent.name)
            return "convex_hull", decompose_collision(mesh, out_path, method="convex_hull")
        raise


def resolve_method(requested: str) -> CollisionMethod:
    have_coacd = coacd_available()
    if requested == "coacd":
        if not have_coacd:
            raise DecompositionUnavailable(
                "--collision coacd requested but the `coacd` package is not importable in this venv. "
                "Use --collision auto (falls back to convex hulls) or install coacd in a separate env."
            )
        return "coacd"
    if requested == "convex_hull":
        return "convex_hull"
    chosen: CollisionMethod = "coacd" if have_coacd else "convex_hull"
    logger.info("collision method 'auto' resolved to %s (coacd_available=%s)", chosen, have_coacd)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        choices=(GSO_SOURCE, YCB_SOURCE),
        default=GSO_SOURCE,
        help="Which collection to import. 'gso' downloads from the Fuel API; 'ycb' reads the YCB set ManiSkill has "
        "already downloaded (no network). Both write the same manifest, with per-entry source/license/attribution.",
    )
    parser.add_argument("--limit", type=int, default=60, help="Max models to attempt (bounded disk/time).")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many models in the listing (batch the ~1030).")
    parser.add_argument(
        "--out", type=Path, default=None, help="Manifest path (default: <cache-dir>/gso_manifest.json)."
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="Asset cache root (default: ~/.cache/sweep_bench/gso)."
    )
    parser.add_argument("--collision", choices=("auto", "coacd", "convex_hull"), default="auto")
    parser.add_argument(
        "--density", type=float, default=1000.0, help="Rider density kg/m^3 -> mass (catalog default 1000)."
    )
    parser.add_argument(
        "--friction", type=float, default=None, help="Rider material friction; omitted = SAPIEN default."
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Resume: merge into an existing manifest instead of replacing it, and skip every model it already "
        "judged (imported or rejected) before downloading. Use this for every batch of a bulk run.",
    )
    parser.add_argument(
        "--retry-rejected",
        metavar="SUBSTRING",
        default=None,
        help="Before resuming, drop the manifest's rejections whose reason contains SUBSTRING, so those models are "
        "downloaded and judged again. This is how a fixed importer bug un-rejects its victims: the resume skip is "
        "keyed on the recorded judgement, so without this they stay rejected forever. Requires --merge.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.retry_rejected and not args.merge:
        parser.error("--retry-rejected only makes sense with --merge (there is no manifest to clear otherwise)")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cache_dir: Path = args.cache_dir or default_cache_dir()
    out_path: Path = args.out or default_manifest_path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        method = resolve_method(args.collision)
    except DecompositionUnavailable as e:
        parser.error(str(e))

    source: str = args.source
    models_dir = ycb_models_dir()
    if source == YCB_SOURCE:
        logger.info("Listing YCB models already downloaded by ManiSkill in %s", models_dir)
        try:
            models = list_ycb_models(models_dir)
        except RuntimeError as e:
            parser.error(str(e))
        models = models[args.offset : args.offset + args.limit]
    else:
        logger.info("Listing up to %d GSO models (offset %d) from %s", args.limit, args.offset, _FUEL_BASE)
        models = list_models(args.limit, args.offset)
    logger.info("Fetched %d model records; processing with collision=%s into %s", len(models), method, cache_dir)

    entries: list[ImportedMeshEntry] = []
    rejections: list[RejectedModel] = []
    seen_ids: set[str] = set()
    judged_ids: set[str] = set()
    pending_revet: tuple[str, ...] = ()
    n_cleared = 0
    if args.merge and out_path.is_file():
        existing = GsoManifest.from_dict(json.loads(out_path.read_text(encoding="utf-8")))
        entries.extend(existing.entries)
        pending_revet = existing.pending_revet
        kept = list(existing.rejections)
        if args.retry_rejected:
            batch_ids = {slug_for(str(m.get("name", "")), prefix=source) for m in models}

            def _retryable(r: RejectedModel) -> bool:
                return args.retry_rejected in r.reason and r.id in batch_ids

            cleared = [r for r in kept if _retryable(r)]
            out_of_batch = sum(1 for r in kept if args.retry_rejected in r.reason and r.id not in batch_ids)
            kept = [r for r in kept if not _retryable(r)]
            n_cleared = len(cleared)
            logger.info("Clearing %d rejection(s) whose reason contains %r", n_cleared, args.retry_rejected)
            for r in cleared:
                logger.info("  re-judging %s (was rejected: %s)", r.name, r.reason)
            if out_of_batch:
                logger.warning(
                    "%d matching rejection(s) are outside this run's source/batch and were KEPT, not retried; "
                    "widen --limit/--offset or rerun with the other --source to re-judge them",
                    out_of_batch,
                )
        rejections.extend(kept)
        seen_ids.update(e.id for e in existing.entries)
        judged_ids.update(r.id for r in rejections)
        logger.info(
            "Resuming: merging into an existing manifest with %d entries and %d rejections",
            len(entries),
            len(rejections),
        )

    n_license_skip = 0
    n_resume_skip = 0
    n_new = 0
    retryable: list[tuple[str, str]] = []

    def flush() -> None:
        manifest = GsoManifest(
            schema_version=1,
            source=",".join(sorted({e.source for e in entries} | {source})),
            license=GSO_LICENSE,
            cache_dir=str(cache_dir.resolve()),
            entries=tuple(entries),
            rejections=tuple(rejections),
            pending_revet=pending_revet,
        )
        write_json_atomic(out_path, manifest.to_dict())

    def reject(name: str, reason: str) -> None:
        logger.warning("REJECT %s: %s", name, reason)
        rejections.append(RejectedModel(name=name, id=slug_for(name, prefix=source), reason=reason))

    for i, model in enumerate(models, start=1):
        name = str(model.get("name", f"<model {i}>"))
        model_id = slug_for(name, prefix=source)
        if model_id in seen_ids or model_id in judged_ids:
            n_resume_skip += 1
            logger.debug("skip already-judged %s", name)
            continue
        if not is_cc_by_4(model):
            n_license_skip += 1
            reject(name, f"license '{model.get('license_name')}' is not CC-BY 4.0")
            judged_ids.add(model_id)
            flush()
            continue
        try:
            if source == YCB_SOURCE:
                entry = process_ycb_model(
                    model,
                    models_dir=models_dir,
                    density=args.density,
                    friction=args.friction,
                )
            else:
                entry = process_model(
                    model,
                    cache_dir=cache_dir,
                    method=method,
                    density=args.density,
                    friction=args.friction,
                    allow_hull_fallback=True,
                )
        except MeshRejected as e:
            reject(name, str(e))
            judged_ids.add(model_id)
            flush()
            continue
        except (urllib.error.URLError, RuntimeError, OSError, zipfile.BadZipFile) as e:
            reason = f"download/extract error: {e}"
            logger.warning("RETRY-LATER %s: %s", name, reason)
            retryable.append((name, reason))
            continue
        except Exception as e:  # noqa: BLE001 — one bad mesh must never kill a ~1000-model run
            reject(name, f"{type(e).__name__} during processing: {e}")
            judged_ids.add(model_id)
            flush()
            continue
        seen_ids.add(entry.id)
        entries.append(entry)
        n_new += 1
        flush()
        logger.info(
            "OK %-40s scale=%.3f bbox=(%.3f,%.3f,%.3f) mass=%.3f parts=%d method=%s",
            entry.id,
            entry.scale,
            *entry.bbox,
            entry.mass,
            entry.collision_parts,
            entry.collision_method,
        )

    flush()

    print(f"\n==== {source.upper()} import summary ====")
    print(f"listed:            {len(models)}")
    print(f"cleared rejections:{n_cleared} (--retry-rejected, re-judged this run)")
    print(f"skipped (resume):  {n_resume_skip}")
    print(f"license-skipped:   {n_license_skip}")
    print(f"newly imported:    {n_new}")
    print(f"retry later:       {len(retryable)} (transient download/extract failures, NOT recorded)")
    print(f"rejected (total):  {len(rejections)} (manifest total, incl. merged)")
    print(f"accepted:          {len(entries)} (manifest total, incl. merged)")
    print(f"manifest:          {out_path}")
    print(f"cache dir:         {cache_dir}")
    if retryable:
        print("\n-- transient failures, re-run to retry (name: reason) --")
        for name, reason in retryable:
            print(f"  {name}: {reason}")
    if rejections:
        print("\n-- rejections (name: reason) --")
        for rejected in rejections:
            print(f"  {rejected.name}: {rejected.reason}")


if __name__ == "__main__":
    main()
