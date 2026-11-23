from __future__ import annotations

import functools
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

import sapien.render

MapRole = Literal["color", "normal", "roughness", "metallic"]
_REQUIRED_ROLES: tuple[MapRole, ...] = ("color", "normal", "roughness")
_SCHEMA_VERSION = 1
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "table_materials.json"
_MIPMAP_LEVELS = 11


@dataclass(frozen=True)
class MapFile:
    role: MapRole
    sha256: str
    size: int
    url: str | None
    member: str | None


@dataclass(frozen=True)
class TableMaterial:
    id: str
    source: str
    category: str
    name: str
    page: str
    authors: tuple[str, ...]
    license: str
    license_url: str
    tile_m: float
    archive: str | None
    maps: tuple[MapFile, ...]

    def path(self, role: MapRole, cache_dir: Path) -> Path:
        return cache_dir / self.id / f"{role}.jpg"


@dataclass(frozen=True)
class TableMaterialManifest:
    default: str
    materials: tuple[TableMaterial, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(m.id for m in self.materials)

    def get(self, material_id: str) -> TableMaterial:
        for material in self.materials:
            if material.id == material_id:
                return material
        raise KeyError(f"unknown table material {material_id!r}; the manifest has {list(self.ids)}")


def default_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "sweep_bench" / "table_materials"


def _field[T](row: dict[str, object], key: str, kind: type[T], where: str) -> T:
    value = row.get(key)
    if not isinstance(value, kind):
        raise ValueError(f"{where}: {key!r} is {value!r}, expected {kind.__name__}")
    return value


def _optional_str(row: dict[str, object], key: str, where: str) -> str | None:
    value = row.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{where}: {key!r} is {value!r}, expected a string or null")
    return value


def _map_file(role: str, row: object, where: str) -> MapFile:
    if role not in get_args(MapRole) or not isinstance(row, dict):
        raise ValueError(f"{where}: map {role!r} is not one of {get_args(MapRole)} with an object value")
    spec = cast("dict[str, object]", row)
    url, member = _optional_str(spec, "url", where), _optional_str(spec, "member", where)
    if (url is None) == (member is None):
        raise ValueError(f"{where}: map {role!r} needs exactly one of 'url' and 'member'")
    return MapFile(
        role=cast("MapRole", role),
        sha256=_field(spec, "sha256", str, where),
        size=_field(spec, "bytes", int, where),
        url=url,
        member=member,
    )


def _material(row: object, index: int) -> TableMaterial:
    if not isinstance(row, dict):
        raise ValueError(f"materials[{index}] is not an object")
    spec = cast("dict[str, object]", row)
    where = f"materials[{index}] ({spec.get('id')!r})"
    maps = _field(spec, "maps", dict, where)
    files = tuple(_map_file(str(role), value, where) for role, value in maps.items())
    missing = [r for r in _REQUIRED_ROLES if r not in {f.role for f in files}]
    archive = _optional_str(spec, "archive", where)
    if missing:
        raise ValueError(f"{where}: missing maps {missing}")
    if any((f.member is not None) != (archive is not None) for f in files):
        raise ValueError(f"{where}: archive members need an 'archive', and URL maps must not have one")
    authors = _field(spec, "authors", list, where)
    tile_m = _field(spec, "tile_m", float, where)
    if tile_m <= 0:
        raise ValueError(f"{where}: tile_m must be positive, got {tile_m}")
    return TableMaterial(
        id=_field(spec, "id", str, where),
        source=_field(spec, "source", str, where),
        category=_field(spec, "category", str, where),
        name=_field(spec, "name", str, where),
        page=_field(spec, "page", str, where),
        authors=tuple(str(a) for a in authors),
        license=_field(spec, "license", str, where),
        license_url=_field(spec, "license_url", str, where),
        tile_m=tile_m,
        archive=archive,
        maps=files,
    )


@functools.cache
def load_manifest(path: Path = MANIFEST_PATH) -> TableMaterialManifest:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"{path}: not a schema-{_SCHEMA_VERSION} table material manifest")
    doc = cast("dict[str, object]", raw)
    rows = _field(doc, "materials", list, str(path))
    manifest = TableMaterialManifest(
        default=_field(doc, "default", str, str(path)),
        materials=tuple(_material(row, i) for i, row in enumerate(rows)),
    )
    if len(set(manifest.ids)) != len(manifest.ids):
        raise ValueError(f"{path}: duplicate material ids")
    manifest.get(manifest.default)
    return manifest


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_problems(material: TableMaterial, cache_dir: Path) -> list[str]:
    problems: list[str] = []
    for spec in material.maps:
        path = material.path(spec.role, cache_dir)
        if not path.is_file():
            problems.append(f"{path} is missing")
        elif sha256_file(path) != spec.sha256:
            problems.append(f"{path} does not match its sha256 {spec.sha256}")
    return problems


@functools.cache
def render_material(material_id: str, cache_dir: Path | None = None) -> sapien.render.RenderMaterial:
    root = cache_dir if cache_dir is not None else default_cache_dir()
    material = load_manifest().get(material_id)
    problems = cache_problems(material, root)
    if problems:
        raise FileNotFoundError(
            f"table material {material_id!r} is not in the cache: {'; '.join(problems)}. "
            f"Run `python scripts/fetch_table_materials.py`."
        )

    def texture(role: MapRole) -> sapien.render.RenderTexture2D:
        return sapien.render.RenderTexture2D(
            str(material.path(role, root)), mipmap_levels=_MIPMAP_LEVELS, srgb=role == "color"
        )

    roles = {f.role for f in material.maps}
    mat = sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0], roughness=1.0, metallic=0.0)
    mat.base_color_texture = texture("color")
    mat.normal_texture = texture("normal")
    mat.roughness_texture = texture("roughness")
    if "metallic" in roles:
        mat.metallic = 1.0
        mat.metallic_texture = texture("metallic")
    return mat
