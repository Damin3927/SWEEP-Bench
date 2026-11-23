from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

    from sweep_bench.data.schemas import BenchSchema


class SchemaStamp(TypedDict):
    name: str
    camera_rig: str


INFO_KEY = "bench_schema"

_INFO_RELPATH = "meta/info.json"


def schema_stamp(schema: BenchSchema) -> SchemaStamp:
    return {"name": schema.name, "camera_rig": schema.camera_rig}


def load_info(dataset_root: Path) -> dict[str, object] | None:
    info_path = dataset_root / _INFO_RELPATH
    if not info_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{info_path} could not be read as JSON, so its provenance is unknown: {exc}") from exc
    fields = as_json_object(info)
    if fields is None:
        raise ValueError(
            f"{info_path} holds a JSON {type(info).__name__} at its top level, but lerobot writes an "
            f"object there. The file has been rewritten or truncated, so this dataset's provenance "
            f"is damaged, not merely unstamped."
        )
    return fields


def as_json_object(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


def read_stamp(dataset_root: Path) -> SchemaStamp | None:
    info_path = dataset_root / _INFO_RELPATH
    info = load_info(dataset_root)
    if info is None:
        return None
    if INFO_KEY not in info:
        return None
    stamp = as_json_object(info[INFO_KEY])
    if stamp is None:
        raise ValueError(
            f"{info_path} has a {INFO_KEY!r} entry of type {type(info[INFO_KEY]).__name__}, expected an object."
        )
    name = stamp.get("name")
    camera_rig = stamp.get("camera_rig")
    if not isinstance(name, str) or not isinstance(camera_rig, str):
        missing = [k for k in SchemaStamp.__required_keys__ if not isinstance(stamp.get(k), str)]
        raise ValueError(
            f"{info_path} has a malformed {INFO_KEY!r} entry: {sorted(missing)} missing or not a string "
            f"(got keys {sorted(stamp)}). This is damaged provenance, not a pre-stamp dataset."
        )
    return {"name": name, "camera_rig": camera_rig}


def read_schema_name(dataset_root: Path) -> str | None:
    stamp = read_stamp(dataset_root)
    return None if stamp is None else stamp["name"]


def check_schema_name(dataset_root: Path, expected: str) -> str | None:
    found = read_schema_name(dataset_root)
    if found is None:
        warnings.warn(
            f"{dataset_root} carries no bench schema stamp, so its camera vintage cannot be "
            f"verified against the expected {expected!r}; re-collect it with scripts/collect.py if "
            f"the camera geometry matters here.",
            stacklevel=2,
        )
        return None
    if found != expected:
        raise ValueError(
            f"{dataset_root} was collected under schema {found!r}, but this consumer expects "
            f"{expected!r}. These vintages differ in camera geometry (see "
            f"`sweep_bench/data/vintage.py`) and must not be mixed; re-collect the "
            f"source, or point this at a {expected!r} dataset."
        )
    return found
