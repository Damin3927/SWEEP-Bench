from __future__ import annotations

import functools
import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from sweep_bench.assets.catalog import ObjectSpec, get_objects
from sweep_bench.assets.mesh_import import default_manifest_path, load_located_entries, load_manifest
from sweep_bench.assets.procedural import PROCEDURAL_SPECS

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from sweep_bench.assets.catalog import ObjectRole
    from sweep_bench.assets.mesh_import import ImportedMeshEntry

logger = logging.getLogger(__name__)

_VETTED_SCHEMA_VERSION = 1

_VETTED_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "vetted_objects.json"


def _as_mapping(value: object, what: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{_VETTED_MANIFEST_PATH}: {what} must be a JSON object")
    return {str(k): v for k, v in value.items()}


def _as_records(value: object, what: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"{_VETTED_MANIFEST_PATH}: {what} must be a list")
    return [_as_mapping(item, f"{what} item") for item in value]


@functools.cache
def _manifest() -> dict[str, object]:
    with _VETTED_MANIFEST_PATH.open(encoding="utf-8") as f:
        manifest = _as_mapping(json.load(f), "manifest")
    version = manifest.get("schema_version")
    if version != _VETTED_SCHEMA_VERSION:
        raise ValueError(
            f"{_VETTED_MANIFEST_PATH}: unsupported schema_version {version!r} (expected {_VETTED_SCHEMA_VERSION})"
        )
    return manifest


def _entries() -> list[dict[str, object]]:
    return _as_records(_manifest()["entries"], "entries")


@functools.cache
def task_local_ids() -> tuple[str, ...]:
    section = _manifest().get("task_local")
    if section is None:
        return ()
    return tuple(str(e["id"]) for e in _as_records(_as_mapping(section, "task_local")["entries"], "task_local.entries"))


@functools.cache
def registered_ids() -> frozenset[str]:
    return frozenset({str(e["id"]) for e in _entries()} | set(task_local_ids()))


@functools.cache
def committed_pass_ids(role: ObjectRole | None = None) -> tuple[str, ...]:
    return tuple(
        str(e["id"]) for e in _entries() if e.get("verdict") == "pass" and (role is None or e.get("role") == role)
    )


@functools.cache
def vetted_pass_ids(role: ObjectRole | None = None) -> tuple[str, ...]:
    pending = _pending_revet_ids(None)
    return tuple(object_id for object_id in committed_pass_ids(role) if object_id not in pending)


_MESH_ID_PREFIXES = ("gso_", "ycb_", "robocasa_", "polyhaven_")

PARTIAL_CACHE_ENV = "SWEEP_ALLOW_PARTIAL_OBJECT_CACHE"


class ResolvedCatalog(tuple[ObjectSpec, ...]):
    unservable: tuple[str, ...]
    gso_manifest_path: str | None

    def __new__(
        cls, specs: Iterable[ObjectSpec], unservable: tuple[str, ...] = (), gso_manifest_path: str | None = None
    ) -> ResolvedCatalog:
        catalog = super().__new__(cls, specs)
        catalog.unservable = unservable
        catalog.gso_manifest_path = gso_manifest_path
        return catalog


@functools.cache
def _pending_revet_ids(manifest_path: str | None) -> frozenset[str]:
    path = Path(manifest_path) if manifest_path is not None else default_manifest_path()
    if not path.is_file():
        return frozenset()
    pending = frozenset(load_manifest(path).pending_revet)
    if pending:
        logger.warning(
            "%d object(s) in %s are marked pending_revet and are WITHHELD from the catalog until their new "
            "verdicts are published (re-vet them with scripts/vet_objects.py --only <ids>, then publish with "
            "scripts/merge_vetted_results.py): %s",
            len(pending),
            path,
            ", ".join(sorted(pending)[:5]) + (", ..." if len(pending) > 5 else ""),
        )
    return pending


@functools.cache
def _mesh_entries(manifest_path: str | None) -> dict[str, ImportedMeshEntry]:
    path = Path(manifest_path) if manifest_path is not None else default_manifest_path()
    if not path.is_file():
        logger.warning(
            "imported-mesh manifest not found at %s; %s catalog ids cannot be served on this machine "
            "(re-run scripts/import_objects.py to rebuild the asset cache)",
            path,
            "/".join(f"{p}*" for p in _MESH_ID_PREFIXES),
        )
        return {}
    pending = _pending_revet_ids(manifest_path)
    return {entry.id: entry for entry in load_located_entries(path) if entry.id not in pending}


@functools.cache
def _mesh_spec_map(manifest_path: str | None) -> dict[str, ObjectSpec]:
    return {i: entry.to_spec() for i, entry in _mesh_entries(manifest_path).items() if entry.has_mesh_files()}


@functools.cache
def _unserved_mesh_prefixes(manifest_path: str | None) -> tuple[str, ...]:
    entries = _mesh_entries(manifest_path)
    unserved = tuple(p for p in _MESH_ID_PREFIXES if not any(i.startswith(p) for i in entries))
    if entries:
        for prefix in unserved:
            logger.warning(
                "imported-mesh manifest at %s holds no %s* objects; those catalog ids cannot be served on this "
                "machine (re-run scripts/import_objects.py --source %s to add them)",
                manifest_path if manifest_path is not None else default_manifest_path(),
                prefix,
                prefix.rstrip("_"),
            )
    return unserved


def resolve_object_specs(
    ids: Sequence[str],
    *,
    extra: Sequence[ObjectSpec] = (),
    gso_manifest_path: str | None = None,
) -> ResolvedCatalog:
    by_id: dict[str, ObjectSpec] = {}
    for spec in (*extra, *get_objects(), *PROCEDURAL_SPECS):
        by_id.setdefault(spec.id, spec)
    entries = _mesh_entries(gso_manifest_path)
    mesh = _mesh_spec_map(gso_manifest_path)
    unserved = _unserved_mesh_prefixes(gso_manifest_path)
    pending = _pending_revet_ids(gso_manifest_path)

    out: list[ObjectSpec] = []
    unservable: list[str] = []
    for object_id in ids:
        if object_id in by_id:
            out.append(by_id[object_id])
        elif object_id in mesh:
            out.append(mesh[object_id])
        elif object_id in pending:
            raise ValueError(
                f"catalog object {object_id!r} is marked pending_revet: its geometry was rewritten and the "
                "verdict measured on the old geometry is not evidence about it. Finish the re-vet "
                "(scripts/vet_objects.py --only <id>, then scripts/merge_vetted_results.py --apply) "
                "before using it."
            )
        elif object_id in entries or object_id.startswith(unserved) or object_id in committed_pass_ids():
            unservable.append(object_id)
        else:
            raise ValueError(f"Unknown catalog object id {object_id!r}")

    if unservable:
        logger.warning(
            "imported-mesh manifest at %s: %d of the %d requested mesh ids cannot be served by this machine's "
            "cache (e.g. %s) and were skipped, leaving %d objects here; an env over this catalog refuses to "
            "build until the cache serves them (or %s=1) -- re-run scripts/import_objects.py --merge in "
            "--limit/--offset batches until the cache holds every vetted mesh",
            _manifest_label(gso_manifest_path),
            len(unservable),
            len(unservable) + sum(1 for i in ids if i in mesh),
            ", ".join(unservable[:3]),
            len(out),
            PARTIAL_CACHE_ENV,
        )
    return ResolvedCatalog(out, tuple(unservable), gso_manifest_path)


def _manifest_label(gso_manifest_path: str | None) -> str:
    return str(gso_manifest_path if gso_manifest_path is not None else default_manifest_path())


def require_servable_catalogs(env_cls: type) -> None:
    if os.environ.get(PARTIAL_CACHE_ENV) == "1":
        return
    short: dict[str, ResolvedCatalog] = {}
    for klass in env_cls.__mro__:
        for name, value in vars(klass).items():
            if name not in short and isinstance(value, ResolvedCatalog) and value.unservable:
                short[name] = value
    if not short:
        return
    reasons = []
    for name, catalog in short.items():
        entries = _mesh_entries(catalog.gso_manifest_path)
        not_held = [i for i in catalog.unservable if i not in entries]
        files_gone = [i for i in catalog.unservable if i in entries]
        if not_held:
            reasons.append(
                f"{name} ids not in the manifest at {_manifest_label(catalog.gso_manifest_path)}: {', '.join(not_held)}"
            )
        if files_gone:
            first = entries[files_gone[0]]
            gone = next(
                (f for f in (first.visual_file, first.collision_file) if not Path(f).is_file()), first.visual_file
            )
            reasons.append(f"{name} ids whose mesh files are missing (e.g. {gone}): {', '.join(files_gone)}")
    missing = sum(len(catalog.unservable) for catalog in short.values())
    raise ValueError(
        f"{env_cls.__name__} would draw from a smaller object pool than the family was screened on: this "
        f"machine's asset cache cannot serve {missing} of its catalog ids. {'; '.join(reasons)}. Complete the "
        "cache (scripts/merge_mesh_manifests.py --check-only names the shortfall), or set "
        f"{PARTIAL_CACHE_ENV}=1 to build over the objects this cache serves."
    )


def display_name(spec: ObjectSpec) -> str:
    if spec.id.startswith("ycb_"):
        body = spec.id[len("ycb_") :]
        return re.sub(r"^\d+[a-z]?_", "", body).replace("_", " ")
    if spec.id.startswith("gso_"):
        return spec.id[len("gso_") :].replace("_", " ")
    if spec.id.startswith("proc_") and len(spec.tags) >= 2:
        return f"{spec.tags[-1]} {spec.tags[1]}"
    return spec.id
