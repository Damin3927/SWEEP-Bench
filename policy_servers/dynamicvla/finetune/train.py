import functools
import importlib.util
import inspect
import io
import json
import logging
import math
import netrc
import os
import pathlib
import subprocess
import sys
import tempfile
import types
from typing import TYPE_CHECKING, NamedTuple

import conv_state
import numpy as np
import speed_token
from conv_state import (
    check_complete,
    family_ranges_path,
    parse_family_ranges,
    validate_convention,
    write_text_atomic,
)

if TYPE_CHECKING:
    import torch

_SCHEMA_INFO_KEY = "bench_schema"
_EXPECTED_SCHEMA = "droid_panda_v3"
_SCHEMA_REQUIRED_FIELDS = ("name", "camera_rig")

EULER_RX_WRAPPED = conv_state.EULER_RX_WRAPPED
EULER_XZ_WRAPPED = conv_state.EULER_XZ_WRAPPED

_UPSTREAM = pathlib.Path(__file__).resolve().parent.parent / "upstream"
sys.path.insert(0, str(_UPSTREAM))


def _family_ranges(dataset_dir: pathlib.Path, total: int) -> dict[str, tuple[int, int]] | None:
    path = family_ranges_path(dataset_dir)
    if path is None:
        return None
    try:
        return parse_family_ranges(path.read_text(), total)
    except (ValueError, OSError) as exc:
        logging.warning("[bench] %s is not a usable family manifest (%s); falling back to the global split", path, exc)
        return None


def _stratified_split(families: dict[str, tuple[int, int]], n_test: int, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    names = sorted(families)
    capacity = [families[n][1] - families[n][0] for n in names]
    quota = _allocate_holdout(capacity, n_test, rng)

    test: list[int] = []
    for name, take in zip(names, quota, strict=True):
        lo, hi = families[name]
        if take > 0:
            test.extend(int(e) for e in rng.permutation(np.arange(lo, hi + 1))[:take])

    test_set = set(test)
    train = [e for lo, hi in families.values() for e in range(lo, hi + 1) if e not in test_set]
    return sorted(train), sorted(test_set)


def _allocate_holdout(capacity: list[int], n_test: int, rng) -> list[int]:
    quota = [0] * len(capacity)
    remaining = n_test
    while remaining > 0:
        eligible = [i for i in range(len(capacity)) if quota[i] < capacity[i]]
        if not eligible:
            break
        share = remaining // len(eligible)
        if share == 0:
            for i in rng.permutation(eligible)[:remaining]:
                quota[int(i)] += 1
            break
        for i in eligible:
            take = min(share, capacity[i] - quota[i])
            quota[i] += take
            remaining -= take
    return quota


def _episode_split(
    repo_id: str, root, fraction: float, seed: int, stratify: bool = True
) -> tuple[list[int], list[int]]:
    import lerobot.constants
    import lerobot.datasets.lerobot_dataset

    resolved = pathlib.Path(root) if root else lerobot.constants.HF_LEROBOT_HOME / repo_id
    meta = lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata(repo_id, resolved)
    total = int(meta.total_episodes)

    if total < 2:
        raise RuntimeError(
            f"{repo_id} has {total} episode(s); DynamicVLA needs at least 2, because "
            "upstream builds both a train and a test loader"
        )
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise RuntimeError(
            f"BENCH_TEST_FRACTION must be a finite value in (0, 1), got {fraction}. Upstream builds both "
            f"a train and a test loader, so neither side may be empty -- and the clamp would otherwise "
            f"turn this into a split that is not the one the log names."
        )
    n_test = min(max(1, int(round(total * fraction))), total - 1)

    families = _family_ranges(resolved, total) if stratify else None
    if families:
        train, test = _stratified_split(families, n_test, seed)
    if families and not test:
        logging.warning(
            "[bench] %s describes %d families averaging %.1f episodes each, too few to hold any out "
            "without emptying a family of training data; falling back to the global split. The "
            "validation loss will not be family-balanced.",
            resolved,
            len(families),
            total / len(families),
        )
        families = None
    if families:
        covered = sum(1 for lo, hi in families.values() if any(lo <= e <= hi for e in test))
        logging.info(
            "[bench] stratified holdout over %d families from %s: %d/%d families represented",
            len(families),
            family_ranges_path(resolved) or resolved,
            covered,
            len(families),
        )
        if len(test) != n_test:
            logging.warning(
                "[bench] the per-family cap reduced the holdout from %d episodes to %d: at "
                "BENCH_TEST_FRACTION=%.4f some families would otherwise have had no training episodes left.",
                n_test,
                len(test),
                fraction,
            )
        held = set(test)
        unrepresented = {n for n, (lo, hi) in families.items() if not any(e in held for e in range(lo, hi + 1))}
        impossible = {n for n in unrepresented if families[n][1] == families[n][0]}
        starved = unrepresented - impossible
        if starved:
            representable = sum(1 for lo, hi in families.values() if hi > lo)
            logging.warning(
                "[bench] BENCH_TEST_FRACTION=%.4f yields %d held-out episodes, so %d of %d families get "
                "no validation episode. Raise the fraction to at least %.4f (%d episodes) for one each.",
                fraction,
                n_test,
                len(starved),
                len(families),
                representable / total,
                representable,
            )
        if impossible:
            shown = sorted(impossible)
            logging.warning(
                "[bench] %s hold%s a single episode and can NEVER be represented in the holdout at any "
                "fraction, since holding it out would leave nothing to train on: %s%s. Raising "
                "BENCH_TEST_FRACTION will not change this; collect more episodes for them, or accept "
                "that the validation loss does not cover them.",
                "1 family" if len(impossible) == 1 else f"{len(impossible)} families",
                "s" if len(impossible) == 1 else "",
                ", ".join(shown[:5]),
                "" if len(shown) <= 5 else f", ... ({len(shown) - 5} more)",
            )
        return train, test

    n_tasks = len(getattr(meta, "tasks", ()) or ())
    if stratify and n_tasks > n_test:
        logging.warning(
            "[bench] %s has %d distinct task strings but no usable %s, so the %d held-out episodes are "
            "drawn globally and cannot cover them all. On a merged multi-family set this silently "
            "validates on a subset (~62 of 100 families at this size). Convert with a converter that "
            "carries the family manifest forward, or set BENCH_SPLIT_STRATIFY=0 to accept this.",
            repo_id,
            n_tasks,
            "ranges.json",
            n_test,
        )

    order = np.random.default_rng(seed).permutation(total)
    return sorted(order[n_test:].tolist()), sorted(order[:n_test].tolist())


def patch_split() -> None:
    fraction = float(os.environ.get("BENCH_TEST_FRACTION", "0.1"))
    seed = int(os.environ.get("BENCH_SPLIT_SEED", "0"))
    stratify = os.environ.get("BENCH_SPLIT_STRATIFY", "1") not in ("0", "false", "False")

    import utils.datasets

    original = utils.datasets.LeRobotDataset.__init__
    signature = inspect.signature(original)
    for required in ("repo_id", "root", "split", "episodes"):
        if required not in signature.parameters:
            raise RuntimeError(
                f"upstream LeRobotDataset.__init__ has no {required!r} parameter; "
                "the split patch needs re-checking against the pinned submodule"
            )

    cache: dict[tuple[str, str], tuple[list[int], list[int]]] = {}

    @functools.wraps(original)
    def patched(self, *args, **kwargs):  # noqa: C901 - one wrapper, read top to bottom
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if bound.arguments.get("episodes") is None:
            repo_id = bound.arguments["repo_id"]
            root = bound.arguments.get("root")
            key = (str(repo_id), str(root))
            if key not in cache:
                cache[key] = _episode_split(repo_id, root, fraction, seed, stratify)
                train, test = cache[key]
                logging.info(
                    "[bench] episode split for %s: %d train / %d test "
                    "(fraction=%.3f seed=%d; upstream's e%%1000 rule would give %d test)",
                    repo_id,
                    len(train),
                    len(test),
                    fraction,
                    seed,
                    len([e for e in range(len(train) + len(test)) if e % 1000 == 0]),
                )
            train, test = cache[key]
            split = bound.arguments.get("split", "train")
            if split == "train":
                bound.arguments["episodes"] = train
            elif split == "test":
                bound.arguments["episodes"] = test
            else:
                raise ValueError(f"Unknown split {split}.")
        result = original(*bound.args, **bound.kwargs)
        if bound.arguments.get("split", "train") == "train":
            _TRAIN_STATS["stats"] = self.stats
        return result

    utils.datasets.LeRobotDataset.__init__ = patched


_TRAIN_STATS: dict[str, object] = {}


def patch_normalization_stats() -> None:
    import utils.helpers

    original = utils.helpers.get_policy

    @functools.wraps(original)
    def patched(policy_cfg, dataset_metadata, *args, **kwargs):
        train_stats = _TRAIN_STATS.get("stats")
        if train_stats is None:
            logging.warning("[bench] no train-split stats captured; using the metadata's")
            return original(policy_cfg, dataset_metadata, *args, **kwargs)
        saved = dataset_metadata.stats
        dataset_metadata.stats = train_stats
        logging.info("[bench] normalizer fitted on the training split only")
        try:
            return original(policy_cfg, dataset_metadata, *args, **kwargs)
        finally:
            dataset_metadata.stats = saved

    utils.helpers.get_policy = patched


def log_parameter_counts() -> None:
    import utils.helpers

    original = utils.helpers.get_n_parameters

    @functools.wraps(original)
    def counted(model, trainable_only: bool = True) -> int:
        n = original(model, trainable_only)
        logging.info("[bench] %s parameters: %d", "trainable" if trainable_only else "total", n)
        return n

    utils.helpers.get_n_parameters = counted


STEM_STAMP = "bench_stem.json"
_STEM_SUFFIX = "vision_model.patch_embedding.0.reparam_conv.weight"


def _config_obs_offsets(argv: list[str]) -> list[int]:
    args = dict(zip(argv, argv[1:], strict=False))
    cfg_path = args.get("-c") or args.get("--cfg")
    if cfg_path is None:
        raise RuntimeError("no -c/--cfg on the command line; cannot read the observation offsets")
    import yaml

    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    return [int(dt) for dt in cfg["DATASET"]["DELTA_TIMESTAMPS"]["observation"]]


def _widen_stem(weight, old_offsets: list[int], new_offsets: list[int]):
    import torch

    missing = sorted(set(old_offsets) - set(new_offsets))
    if missing:
        raise RuntimeError(f"pretrained frame offsets {missing} are not among the new offsets {new_offsets}")
    out = torch.zeros(weight.shape[0], 3 * len(new_offsets), *weight.shape[2:], dtype=weight.dtype)
    for i, dt in enumerate(old_offsets):
        j = new_offsets.index(dt)
        out[:, 3 * j : 3 * j + 3] = weight[:, 3 * i : 3 * i + 3]
    return out


def _fold_stem(weight, n_old: int):
    return weight.reshape(weight.shape[0], n_old, 3, *weight.shape[2:]).sum(dim=1)


def patch_stem(argv: list[str]) -> None:
    import safetensors.torch
    from policies.dynamicvla import modeling_dynamicvla, modeling_vlm_with_expert

    train_stem = os.environ.get("BENCH_TRAIN_STEM", "0") == "1"
    new_offsets = _config_obs_offsets(argv)
    args = dict(zip(argv, argv[1:], strict=False))
    exp = args.get("-e") or args.get("--exp")
    is_rank_zero = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
    run_dir = None if exp is None else pathlib.Path("runs") / "checkpoints" / exp

    policy_cls = modeling_dynamicvla.DynamicVLAPolicy
    original_load = policy_cls._load_as_safetensor.__func__

    def load_mapped(cls, model, state, map_location, strict):
        with tempfile.TemporaryDirectory() as tmp:
            mapped = pathlib.Path(tmp) / "model.safetensors"
            safetensors.torch.save_file(state, str(mapped))
            return original_load(cls, model, str(mapped), map_location, strict)

    def load(cls, model, model_file, map_location, strict):
        state = safetensors.torch.load_file(model_file, device="cpu")
        key = next((k for k in state if k.endswith(_STEM_SUFFIX)), None)
        target = next((v for k, v in model.state_dict().items() if k.endswith(_STEM_SUFFIX)), None)
        if key is None or target is None:
            return original_load(cls, model, model_file, map_location, strict)
        ckpt_cfg_path = pathlib.Path(model_file).parent / "config.json"
        ckpt_cfg = json.loads(ckpt_cfg_path.read_text()) if ckpt_cfg_path.is_file() else {}
        old_offsets = (ckpt_cfg.get("delta_timestamps") or {}).get("observation")
        if old_offsets is None:
            if state[key].shape == target.shape:
                return original_load(cls, model, model_file, map_location, strict)
            raise RuntimeError(f"{ckpt_cfg_path} records no observation offsets; cannot widen the stem")
        old_offsets = [int(dt) for dt in old_offsets]
        if old_offsets == new_offsets:
            return original_load(cls, model, model_file, map_location, strict)
        if state[key].shape == target.shape:
            raise RuntimeError(
                f"{model_file} was trained on offsets {old_offsets} but this run uses {new_offsets}; "
                "the stem shapes match, so the weights would load onto the wrong frames"
            )
        if 3 * len(old_offsets) != state[key].shape[1]:
            raise RuntimeError(f"{model_file}: stem takes {state[key].shape[1]} channels but offsets are {old_offsets}")
        if len(new_offsets) == 1:
            state[key] = _fold_stem(state[key], len(old_offsets))
            logging.info("[bench] folded the stem from offsets %s onto %s", old_offsets, new_offsets)
        else:
            if model.config.freeze_vision_model and not train_stem:
                raise RuntimeError(
                    f"widening the stem to offsets {new_offsets} with a frozen vision model keeps the added frames' "
                    "weights at zero for the whole run; set BENCH_TRAIN_STEM=1 (make TRAIN_STEM=1)"
                )
            state[key] = _widen_stem(state[key], old_offsets, new_offsets)
            logging.info("[bench] widened the stem from offsets %s to %s", old_offsets, new_offsets)
        return load_mapped(cls, model, state, map_location, strict)

    policy_cls._load_as_safetensor = classmethod(load)

    if train_stem:
        expert_cls = modeling_vlm_with_expert.VLMWithExpertModel
        original_freeze = expert_cls._set_requires_grad

        def set_requires_grad(self) -> None:
            original_freeze(self)
            stem = self._get_vision_model(self.get_vlm_model()).patch_embedding[0]
            for param in stem.parameters():
                param.requires_grad = True
            logging.info(
                "[bench] training the vision stem's first conv (%d params)", sum(p.numel() for p in stem.parameters())
            )

        expert_cls._set_requires_grad = set_requires_grad

    if run_dir is None:
        return
    stamp_path = run_dir / STEM_STAMP
    stamp = {"train_stem": train_stem, "obs_offsets": new_offsets}
    if stamp_path.is_file():
        recorded = json.loads(stamp_path.read_text())
        if recorded != stamp:
            raise RuntimeError(
                f"{run_dir} already holds checkpoints stamped {recorded}, but this run is {stamp}. "
                "Use a fresh EXP_NAME."
            )
        return
    unstamped = sorted(run_dir.glob("model*.safetensors"))
    if unstamped:
        raise RuntimeError(
            f"{run_dir} already holds {len(unstamped)} checkpoint(s) without {STEM_STAMP}; their stem setup "
            f"is unknown, so stamping {stamp} could mislabel them. Use a fresh EXP_NAME."
        )
    if is_rank_zero:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(stamp_path, json.dumps(stamp, indent=2))


EULER_STAMP = "bench_euler_convention.json"


def stamp_euler_convention(argv: list[str]) -> None:
    args = dict(zip(argv, argv[1:], strict=False))
    exp = args.get("-e") or args.get("--exp")
    dataset = args.get("-d") or args.get("--dataset")
    if exp is None or dataset is None:
        logging.warning("[bench] no -e/-d on the command line; skipping the euler stamp")
        return

    convention = _dataset_euler_convention(dataset)
    run_dir = pathlib.Path("runs") / "checkpoints" / exp
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = run_dir / EULER_STAMP
    is_rank_zero = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
    if stamp_path.is_file():
        recorded = str(json.loads(stamp_path.read_text())["euler_convention"])
        if recorded != convention:
            raise RuntimeError(
                f"{run_dir} already holds checkpoints stamped euler_convention={recorded!r}, but "
                f"{dataset} is {convention!r}. Rewriting the stamp would relabel every checkpoint already "
                f"in that directory, so the old weights would be served under a convention they were not "
                f"trained on. Use a fresh EXP_NAME, or point this run at a {recorded!r} dataset."
            )
        logging.info("[bench] %s already stamped euler_convention=%s; left unchanged", stamp_path, recorded)
        return
    if not is_rank_zero:
        return
    write_text_atomic(stamp_path, json.dumps({"euler_convention": convention}, indent=2))
    logging.info(
        "[bench] stamped %s with euler_convention=%s (from the dataset %s)",
        stamp_path,
        convention,
        dataset,
    )


def _read_speeds(path: str) -> list[float]:
    try:
        data = json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"BENCH_SPEED_TOKEN={path} is not a readable JSON file: {exc}") from exc
    if (
        not isinstance(data, list)
        or not data
        or not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in data)
    ):
        raise RuntimeError(f"BENCH_SPEED_TOKEN={path} must be a non-empty JSON list of positive speeds")
    return [float(v) for v in data]


def patch_speed_token(argv: list[str]) -> None:
    args = dict(zip(argv, argv[1:], strict=False))
    exp = args.get("-e") or args.get("--exp")
    pretrained = args.get("-p") or args.get("--pretrained")
    run_dir = pathlib.Path("runs") / "checkpoints" / exp if exp else None
    run_stamp = run_dir / speed_token.STAMP if run_dir else None
    pretrained_stamp = speed_token.find_stamp(pathlib.Path(pretrained).resolve()) if pretrained else None
    path = os.environ.get("BENCH_SPEED_TOKEN")
    if not path:
        for stamp in (run_stamp, pretrained_stamp):
            if stamp is not None and stamp.is_file():
                raise RuntimeError(f"{stamp} marks a speed-conditioned checkpoint; set BENCH_SPEED_TOKEN to train it")
        return
    if run_dir is None:
        raise RuntimeError("BENCH_SPEED_TOKEN needs -e on the command line, to stamp the run dir")
    speeds = _read_speeds(path)

    import torch
    import utils.datasets
    from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy

    original_getitem = utils.datasets.LeRobotDataset.__getitem__

    @functools.wraps(original_getitem)
    def getitem(self, idx):
        if self.meta.total_episodes != len(speeds):
            raise RuntimeError(
                f"BENCH_SPEED_TOKEN={path} holds {len(speeds)} speeds but {self.repo_id} has "
                f"{self.meta.total_episodes} episodes"
            )
        item = original_getitem(self, idx)
        ep = self._get_episode_index(idx, self.episode_data_index["from"], self.episode_data_index["to"])
        item[speed_token.SPEED_KEY] = torch.tensor(speeds[self.episodes[ep]], dtype=torch.float32)
        return item

    utils.datasets.LeRobotDataset.__getitem__ = getitem
    speed_token.patch_prepare_state(DynamicVLAPolicy)

    if pretrained_stamp is not None:
        speed_token.read_stamp(pretrained_stamp)
        logging.info(
            "[bench] %s is already speed-conditioned; its slot-%d column is kept", pretrained, speed_token.SPEED_SLOT
        )
    else:
        original_load = DynamicVLAPolicy.from_pretrained.__func__

        def from_pretrained(cls, *a, **kw):
            policy = original_load(cls, *a, **kw)
            if not policy.config.train_state_proj:
                raise RuntimeError("train_state_proj is off, so the zeroed speed column could never learn")
            speed_token.zero_speed_column(policy.model.state_proj)
            logging.info("[bench] zeroed state_proj.weight[:, %d] for the speed token", speed_token.SPEED_SLOT)
            return policy

        DynamicVLAPolicy.from_pretrained = classmethod(from_pretrained)

    if run_stamp.is_file():
        speed_token.read_stamp(run_stamp)
    elif any(run_dir.glob("**/*.safetensors")):
        raise RuntimeError(
            f"{run_dir} already holds checkpoints that are not speed-conditioned; stamping it would relabel "
            "them. Use a fresh EXP_NAME."
        )
    elif int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(run_stamp, json.dumps({"speed_token": speed_token.STAMP_VALUE}, indent=2))
    logging.info(
        "[bench] BENCH_SPEED_TOKEN=%s: %d per-episode speeds, slot %d", path, len(speeds), speed_token.SPEED_SLOT
    )


def check_camera_vintage(argv: list[str]) -> None:
    args = dict(zip(argv, argv[1:], strict=False))
    dataset = args.get("-d") or args.get("--dataset")
    if dataset is None:
        logging.warning("[bench] no -d on the command line; skipping the camera-vintage check")
        return
    info_path = _dataset_root(dataset) / "meta" / "info.json"
    if not info_path.is_file():
        _check_remote_camera_vintage(dataset)
        return
    _validate_camera_stamp(json.loads(info_path.read_text()).get(_SCHEMA_INFO_KEY), source=str(info_path))


def _check_remote_camera_vintage(repo_id: str) -> None:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:
        logging.warning(
            "[bench] %s is not in the local cache and huggingface_hub is unavailable, so its camera "
            "vintage cannot be checked before upstream downloads it; expected %s",
            repo_id,
            _EXPECTED_SCHEMA,
        )
        return
    try:
        path = hf_hub_download(repo_id=repo_id, filename="meta/info.json", repo_type="dataset")
    except EntryNotFoundError:
        logging.warning(
            "[bench] %s has no meta/info.json on the hub, so its camera vintage cannot be verified; expected %s",
            repo_id,
            _EXPECTED_SCHEMA,
        )
        return
    except Exception as exc:  # noqa: BLE001 — an unreachable hub is not a verdict
        logging.warning(
            "[bench] could not read %s's metadata from the hub (%s: %s), so its camera vintage was not "
            "checked; expected %s",
            repo_id,
            type(exc).__name__,
            exc,
            _EXPECTED_SCHEMA,
        )
        return
    _validate_camera_stamp(
        json.loads(pathlib.Path(path).read_text()).get(_SCHEMA_INFO_KEY), source=f"{repo_id} (from the hub)"
    )


def _validate_camera_stamp(stamp: object, *, source: str) -> None:
    if stamp is None:
        logging.warning(
            "[bench] %s carries no %r stamp, so its camera vintage cannot be verified against the expected %s.",
            source,
            _SCHEMA_INFO_KEY,
            _EXPECTED_SCHEMA,
        )
        return
    missing = (
        list(_SCHEMA_REQUIRED_FIELDS)
        if not isinstance(stamp, dict)
        else [k for k in _SCHEMA_REQUIRED_FIELDS if not isinstance(stamp.get(k), str)]
    )
    if missing:
        raise RuntimeError(
            f"{source} has a malformed {_SCHEMA_INFO_KEY!r} entry: {sorted(missing)} missing or not a "
            f"string (got {stamp!r})."
        )
    if stamp["name"] != _EXPECTED_SCHEMA:
        raise RuntimeError(
            f"{source} was collected under camera schema {stamp['name']!r}, but this server serves "
            f"{_EXPECTED_SCHEMA!r} (sweep_bench/policies/ee_dom.py). Those vintages differ in "
            f"camera geometry, so the checkpoint would be evaluated against cameras it never saw; "
            f"re-collect the source, or point DATASET_REPO at a {_EXPECTED_SCHEMA!r} dataset."
        )
    logging.info("[bench] %s camera vintage verified: %s", source, _EXPECTED_SCHEMA)


def _dataset_euler_convention(repo_id: str) -> str:
    root = _dataset_root(repo_id)
    check_complete(root)
    marker = root / "meta" / EULER_STAMP
    if marker.is_file():
        return validate_convention(str(json.loads(marker.read_text())["euler_convention"]), source=str(marker))
    if root.is_dir():
        logging.warning(
            "[bench] %s has no euler-convention marker, so it is read as upstream's DOM convention %s; "
            "convert_droid_dataset.py writes the marker and %s.",
            root,
            EULER_XZ_WRAPPED,
            EULER_RX_WRAPPED,
        )
        return EULER_XZ_WRAPPED
    return _remote_euler_convention(repo_id)


def _remote_euler_convention(repo_id: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError as exc:  # pragma: no cover - the server env ships it
        raise RuntimeError(
            f"{repo_id} is not in the local cache and huggingface_hub is unavailable, so its euler "
            f"convention cannot be resolved before upstream downloads it. Convert or download the "
            f"dataset first."
        ) from exc
    try:
        path = hf_hub_download(repo_id=repo_id, filename=f"meta/{EULER_STAMP}", repo_type="dataset")
    except EntryNotFoundError:
        logging.warning(
            "[bench] %s carries no euler-convention marker on the hub, so it is read as upstream's DOM "
            "convention %s; convert_droid_dataset.py writes the marker and %s.",
            repo_id,
            EULER_XZ_WRAPPED,
            EULER_RX_WRAPPED,
        )
        return EULER_XZ_WRAPPED
    except Exception as exc:
        raise RuntimeError(
            f"{repo_id} is not in the local cache and its euler-convention marker could not be read "
            f"from the hub ({type(exc).__name__}: {exc}). Refusing to guess: stamping the wrong branch "
            f"corrupts every evaluation of the resulting checkpoint without raising."
        ) from exc
    convention = validate_convention(
        str(json.loads(pathlib.Path(path).read_text())["euler_convention"]), source=f"{repo_id} (from the hub)"
    )
    logging.info("[bench] resolved %s's euler convention from the hub: %s", repo_id, convention)
    return convention


def _dataset_root(repo_id: str) -> pathlib.Path:
    root = pathlib.Path(repo_id).expanduser()
    if root.is_absolute():
        return root
    return _lerobot_home() / repo_id


def _lerobot_home() -> pathlib.Path:
    for module in ("lerobot.constants", "lerobot.common.constants"):
        try:
            return pathlib.Path(importlib.import_module(module).HF_LEROBOT_HOME).expanduser()
        except (ImportError, AttributeError):
            continue
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


class _PyAVFrames(NamedTuple):
    data: "torch.Tensor"
    pts_seconds: "torch.Tensor"


class _PyAVVideoDecoder:
    def __init__(self, source: bytes, seek_mode: str = "exact") -> None:
        self._source = source

    def get_frames_at(self, indices: list[int]) -> _PyAVFrames:
        import av
        import torch

        wanted = set(indices)
        decoded: dict[int, tuple[torch.Tensor, float]] = {}
        with av.open(io.BytesIO(self._source)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i in wanted:
                    rgb = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                    decoded[i] = (rgb, float(frame.time))
                    if len(decoded) == len(wanted):
                        break
        missing = wanted - decoded.keys()
        if missing:
            raise IndexError(f"frames {sorted(missing)} are past the end of the video")
        return _PyAVFrames(
            data=torch.stack([decoded[i][0] for i in indices]),
            pts_seconds=torch.tensor([decoded[i][1] for i in indices], dtype=torch.float64),
        )


def install_video_backend() -> None:
    backend = os.environ.get("BENCH_VIDEO_BACKEND", "torchcodec")
    if backend == "torchcodec":
        return
    if backend != "pyav":
        raise ValueError(f"BENCH_VIDEO_BACKEND={backend!r}; expected 'torchcodec' or 'pyav'")
    if "torchcodec" in sys.modules or "utils.datasets" in sys.modules:
        raise RuntimeError("utils.datasets or torchcodec was imported before the pyav backend could replace it")
    decoders = types.ModuleType("torchcodec.decoders")
    decoders.VideoDecoder = _PyAVVideoDecoder  # type: ignore[attr-defined]  # module stub
    package = types.ModuleType("torchcodec")
    package.decoders = decoders  # type: ignore[attr-defined]  # module stub
    sys.modules["torchcodec"] = package
    sys.modules["torchcodec.decoders"] = decoders
    try:
        import utils.datasets  # noqa: F401
    finally:
        del sys.modules["torchcodec"], sys.modules["torchcodec.decoders"]
    logging.info("video backend: pyav (utils.datasets decodes with PyAV)")


def _config_n_workers(argv: list[str]) -> int:
    args = dict(zip(argv, argv[1:], strict=False))
    cfg_path = args.get("-c") or args.get("--cfg")
    if cfg_path is None:
        raise RuntimeError("no -c/--cfg on the command line; cannot read the loader worker count")
    import yaml

    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    return int(cfg["CONST"]["N_WORKERS"])


def patch_affinity(argv: list[str]) -> None:
    import utils.distributed

    needed = _config_n_workers(argv) + 1

    def set_affinity(gpu_id: int | None = None) -> set[int]:
        if gpu_id is None:
            gpu_id = int(os.getenv("LOCAL_RANK", 0))
        allowed = os.sched_getaffinity(0)
        cores = set(utils.distributed.Device(gpu_id).getCpuAffinity()) & allowed
        if len(cores) >= needed:
            os.sched_setaffinity(0, cores)
        else:
            logging.warning(
                "[bench] %d of this job's %d cores are local to GPU %d, fewer than the %d the trainer and its "
                "loader workers need; affinity left unchanged",
                len(cores),
                len(allowed),
                gpu_id,
                needed,
            )
        return os.sched_getaffinity(0)

    utils.distributed.set_affinity = set_affinity


def _code_commit() -> str:
    root = pathlib.Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        return "unknown"
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _wandb_key_available() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        entry = netrc.netrc().authenticators("api.wandb.ai")
    except (OSError, netrc.NetrcParseError):
        return False
    return entry is not None and bool(entry[2])


def patch_wandb(argv: list[str]) -> None:
    args = dict(zip(argv, argv[1:], strict=False))
    cfg_path = args.get("-c") or args.get("--cfg")
    if cfg_path is None:
        return
    import yaml

    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text()).get("WANDB") or {}
    if not cfg.get("ENABLED"):
        return
    import wandb

    tags = [f"commit={_code_commit()}"]
    name = cfg.get("NAME")
    original_init = wandb.init

    def init(*init_args: object, **kwargs: object) -> object:
        if name:
            kwargs["name"] = name
        kwargs["tags"] = tags
        if kwargs.get("mode") == "online" and not _wandb_key_available():
            logging.warning("[bench] no W&B API key on this host; logging offline (wandb sync the run later)")
            kwargs["mode"] = "offline"
        return original_init(*init_args, **kwargs)

    wandb.init = init


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")
    install_video_backend()
    patch_split()
    patch_normalization_stats()
    log_parameter_counts()
    patch_stem(sys.argv)
    check_camera_vintage(sys.argv)
    stamp_euler_convention(sys.argv)
    patch_speed_token(sys.argv)
    patch_affinity(sys.argv)
    patch_wandb(sys.argv)

    run_path = _UPSTREAM / "run.py"
    spec = importlib.util.spec_from_file_location("dynamicvla_upstream_run", run_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {run_path}")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    run.main()


if __name__ == "__main__":
    main()
