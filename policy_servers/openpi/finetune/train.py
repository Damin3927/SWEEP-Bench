from __future__ import annotations

import importlib.util
import pathlib
import sys

import openpi.training.checkpoints  # noqa: F401

import datasets.features.features as _datasets_features

_feature_types = getattr(_datasets_features, "_FEATURE_TYPES", None)
if _feature_types is None:
    raise RuntimeError(
        "datasets.features.features._FEATURE_TYPES is missing — "
        "the `List`->`Sequence` alias must be reworked for this datasets version."
    )
_feature_types.setdefault("List", _datasets_features.Sequence)

_HERE = pathlib.Path(__file__).resolve().parent
_UPSTREAM_TRAIN = _HERE.parent / "upstream" / "scripts" / "train.py"

sys.path.insert(0, str(_HERE))

import bench_configs  # noqa: E402
import grad_accum  # noqa: E402

_BENCH_CAMERA_VINTAGE = bench_configs.check_dataset_camera_vintage()
bench_configs.register()

import openpi.training.config as _config  # noqa: E402

_spec = importlib.util.spec_from_file_location("openpi_upstream_train", _UPSTREAM_TRAIN)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"failed to locate upstream train.py at {_UPSTREAM_TRAIN}")
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)

bench_configs.assert_checkpoint_hook_reachable(_train, openpi.training.checkpoints)
bench_configs.patch_checkpoint_save_limit(openpi.training.checkpoints, bench_configs.SAVE_CONCURRENT_GB)
bench_configs.patch_checkpoint_vintage_stamp(openpi.training.checkpoints, _BENCH_CAMERA_VINTAGE)
bench_configs.patch_checkpoint_history_stamp(openpi.training.checkpoints)
bench_configs.patch_checkpoint_gt_speed_stamp(openpi.training.checkpoints)
bench_configs.patch_checkpoint_sampler_stamp(openpi.training.checkpoints)
bench_configs.patch_checkpoint_action_space_stamp(openpi.training.checkpoints)
bench_configs.patch_resume_to_current_devices(_train, openpi.training.checkpoints)

if __name__ == "__main__":
    _train.init_logging()
    grad_accum.patch_train_step(_train)
    config = _config.cli()
    bench_configs.patch_checkpoint_schedule_stamp(openpi.training.checkpoints, config)
    _train.main(config)
