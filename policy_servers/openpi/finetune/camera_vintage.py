from __future__ import annotations

import functools
import importlib
import inspect
import json
import os
import pathlib

SCHEMA_INFO_KEY = "bench_schema"
EXPECTED_SCHEMA = "droid_panda_v3"
SCHEMA_REQUIRED_FIELDS = ("name", "camera_rig")
SCHEMA_OVERRIDE_ENV = "BENCH_CAMERA_SCHEMA"
OVERRIDE_SCHEMAS = ("droid_rebot_real_v1",)


def expected_schema() -> str:
    override = os.environ.get(SCHEMA_OVERRIDE_ENV)
    if not override:
        return EXPECTED_SCHEMA
    if override not in OVERRIDE_SCHEMAS:
        raise RuntimeError(
            f"{SCHEMA_OVERRIDE_ENV}={override!r} is not one of {OVERRIDE_SCHEMAS}; unset it to train on "
            f"{EXPECTED_SCHEMA!r} data."
        )
    return override


def check_camera_vintage(repo_id: str) -> str | None:
    info_path = dataset_root(repo_id) / "meta" / "info.json"
    if not info_path.is_file():
        return _check_remote_camera_vintage(repo_id)
    return validate_camera_stamp(json.loads(info_path.read_text()).get(SCHEMA_INFO_KEY), source=str(info_path))


def dataset_root(repo_id: str) -> pathlib.Path:
    root = pathlib.Path(repo_id).expanduser()
    if root.is_absolute():
        return root
    return lerobot_home() / repo_id


def lerobot_home() -> pathlib.Path:
    for module in ("lerobot.constants", "lerobot.common.constants"):
        try:
            home = importlib.import_module(module).HF_LEROBOT_HOME
        except (ImportError, AttributeError):
            continue
        return pathlib.Path(home).expanduser()
    return lerobot_home_from_env()


def lerobot_home_from_env() -> pathlib.Path:
    explicit = os.environ.get("HF_LEROBOT_HOME")
    if explicit:
        return pathlib.Path(explicit).expanduser()
    try:
        from huggingface_hub.constants import HF_HOME
    except ImportError:
        HF_HOME = os.environ.get("HF_HOME") or os.path.join(
            os.environ.get("XDG_CACHE_HOME") or "~/.cache", "huggingface"
        )
    return pathlib.Path(HF_HOME).expanduser() / "lerobot"


def _check_remote_camera_vintage(repo_id: str) -> str | None:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:
        _warn(
            f"{repo_id} is not in the local LeRobot cache and huggingface_hub is unavailable, so its "
            f"camera vintage cannot be checked before upstream downloads it; expected {expected_schema()!r}."
        )
        return
    try:
        path = hf_hub_download(repo_id=repo_id, filename="meta/info.json", repo_type="dataset")
    except EntryNotFoundError:
        _warn(
            f"{repo_id} has no meta/info.json on the hub, so its camera vintage cannot be verified; "
            f"expected {expected_schema()!r}."
        )
        return
    except Exception as exc:  # noqa: BLE001 -- an unreachable hub is not a verdict
        _warn(
            f"could not read {repo_id}'s metadata from the hub ({type(exc).__name__}: {exc}), so its "
            f"camera vintage was not checked; expected {expected_schema()!r}."
        )
        return
    return validate_camera_stamp(
        json.loads(pathlib.Path(path).read_text()).get(SCHEMA_INFO_KEY), source=f"{repo_id} (from the hub)"
    )


def validate_camera_stamp(stamp: object, *, source: str) -> str | None:
    expected = expected_schema()
    if stamp is None:
        _warn(
            f"{source} carries no {SCHEMA_INFO_KEY!r} stamp, so its camera vintage cannot be verified "
            f"against the expected {expected!r}."
        )
        return None
    missing = (
        list(SCHEMA_REQUIRED_FIELDS)
        if not isinstance(stamp, dict)
        else [k for k in SCHEMA_REQUIRED_FIELDS if not isinstance(stamp.get(k), str)]
    )
    if missing:
        raise RuntimeError(
            f"{source} has a malformed {SCHEMA_INFO_KEY!r} entry: {sorted(missing)} missing or not a "
            f"string (got {stamp!r})."
        )
    if stamp["name"] != expected:
        raise RuntimeError(
            f"{source} was collected under camera schema {stamp['name']!r}, but this server serves "
            f"{expected!r} (sweep_bench/policies/openpi.py). Those vintages differ in "
            f"camera geometry, so the checkpoint would be evaluated against cameras it never saw; "
            f"re-collect the source, or point BENCH_DATASET_REPO at a {expected!r} dataset."
        )
    print(f"[bench] {source} camera vintage verified: {expected}", flush=True)
    return expected


CHECKPOINT_STAMP = "bench_camera_vintage.json"


def stamp_checkpoint_vintage(
    run_dir: pathlib.Path, schema: str | None, *, existing_steps: tuple[int, ...] = ()
) -> None:
    run_dir = pathlib.Path(run_dir)
    stamp_path = run_dir / CHECKPOINT_STAMP
    if stamp_path.is_file():
        recorded = json.loads(stamp_path.read_text()).get("camera_schema")
        if recorded != schema:
            raise RuntimeError(
                f"{run_dir} already holds checkpoints stamped camera_schema={recorded!r}, but this run "
                f"trains on {schema!r}. Rewriting the stamp would relabel every checkpoint already in "
                f"that directory, so old weights would be served under a vintage they were not trained "
                f"on. Use a fresh --exp-name, or point BENCH_DATASET_REPO at a {recorded!r} dataset."
            )
        return
    if existing_steps:
        raise RuntimeError(
            f"{run_dir} already holds checkpoints (steps {sorted(existing_steps)}) but no "
            f"{CHECKPOINT_STAMP}, so their camera vintage cannot be established from the files, and "
            f"writing this run's {schema!r} over them would label them with a vintage they may not have "
            f"been trained on. Resume under a fresh --exp-name, or pass OVERWRITE=1 to delete the existing "
            f"steps and start a stamped run."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = stamp_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"camera_schema": schema}, indent=2))
    tmp.replace(stamp_path)
    print(f"[bench] stamped {stamp_path} with camera_schema={schema!r}", flush=True)


def patch_checkpoint_vintage_stamp(checkpoints_module, schema: str | None) -> None:
    original = checkpoints_module.initialize_checkpoint_dir
    if "checkpoint_dir" not in inspect.signature(original).parameters:
        raise RuntimeError(
            "upstream initialize_checkpoint_dir has no 'checkpoint_dir' parameter; "
            "the camera-vintage stamp patch needs re-checking against the pinned submodule"
        )

    @functools.wraps(original)
    def patched(checkpoint_dir, **kwargs):
        manager, resuming = original(checkpoint_dir, **kwargs)
        stamp_checkpoint_vintage(checkpoint_dir, schema, existing_steps=tuple(manager.all_steps()))
        return manager, resuming

    checkpoints_module.initialize_checkpoint_dir = patched


def assert_checkpoint_hook_reachable(train_module: object, checkpoints_module: object) -> None:
    bound = getattr(train_module, "_checkpoints", None)
    if bound is not checkpoints_module:
        raise RuntimeError(
            f"upstream's train module no longer resolves `initialize_checkpoint_dir` through the "
            f"`openpi.training.checkpoints` module object (found {bound!r}), so the bench camera-vintage "
            f"hook would never fire and checkpoints would be written unstamped. Re-point the patch at "
            f"whatever upstream binds now -- see `camera_vintage.stamp_checkpoint_vintage`."
        )


def is_remote(ckpt_dir: object) -> bool:
    return "://" in str(ckpt_dir)


def read_checkpoint_vintage(ckpt_dir: pathlib.Path) -> tuple[str | None, object | None, bool]:
    if is_remote(ckpt_dir):
        return _read_remote_checkpoint_vintage(str(ckpt_dir))
    ckpt_dir = pathlib.Path(ckpt_dir)
    for candidate in (ckpt_dir, *ckpt_dir.parents):
        stamp_path = candidate / CHECKPOINT_STAMP
        if stamp_path.is_file():
            recorded = json.loads(stamp_path.read_text()).get("camera_schema")
            return (None if recorded is None else str(recorded)), stamp_path, True
    return None, None, True


def _read_remote_checkpoint_vintage(uri: str) -> tuple[str | None, object | None, bool]:
    try:
        from openpi.shared import download as _download
    except ImportError:
        _warn(f"openpi's download helper is unavailable, so {uri}'s camera vintage was not checked.")
        return None, None, False
    prefix, _, path = uri.partition("://")
    segments = [segment for segment in path.split("/") if segment]
    for depth in range(len(segments), 0, -1):
        candidate = f"{prefix}://{'/'.join(segments[:depth])}/{CHECKPOINT_STAMP}"
        try:
            local = _download.maybe_download(candidate)
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001 -- an unreachable bucket is not a verdict
            _warn(
                f"could not read {candidate} ({type(exc).__name__}: {exc}), so {uri}'s camera vintage was not checked."
            )
            return None, None, False
        recorded = json.loads(pathlib.Path(local).read_text()).get("camera_schema")
        return (None if recorded is None else str(recorded)), candidate, True
    return None, None, True


def check_checkpoint_camera_vintage(ckpt_dir: pathlib.Path, *, expected: str = EXPECTED_SCHEMA) -> None:
    declared, stamp_path, readable = read_checkpoint_vintage(ckpt_dir)
    if not readable:
        return
    if declared == expected:
        print(f"[bench] checkpoint camera vintage {declared} matches the client (per {stamp_path})", flush=True)
        return
    if stamp_path is None:
        source = f"{ckpt_dir} carries no {CHECKPOINT_STAMP}, so its camera vintage is unknown"
    else:
        source = f"{ckpt_dir} was trained under camera schema {declared!r} (per {stamp_path})"
    if expected != EXPECTED_SCHEMA:
        raise RuntimeError(
            f"{source}, but {SCHEMA_OVERRIDE_ENV}={expected!r} says this server is for the {expected!r} client. "
            f"Unset {SCHEMA_OVERRIDE_ENV} to serve a {EXPECTED_SCHEMA!r} checkpoint to the sim client."
        )
    raise RuntimeError(
        f"{source}, but the bench client renders {expected!r} "
        f"(sweep_bench/policies/openpi.py). Serving this pair would score the checkpoint against "
        f"camera geometry it may never have seen. {expected!r} is the only registered camera schema "
        f"(sweep_bench/data/schemas.py); train on a {expected!r} dataset with this repo's openpi "
        f"training, which stamps the run."
    )


def _warn(message: str) -> None:
    print(f"[bench] warning: {message}", flush=True)
