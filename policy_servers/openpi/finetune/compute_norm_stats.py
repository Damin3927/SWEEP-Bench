from __future__ import annotations

import os
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
_UPSTREAM_SCRIPTS = _HERE.parent / "upstream" / "scripts"

sys.path.insert(0, str(_HERE))

import bench_configs  # noqa: E402

bench_configs.check_dataset_camera_vintage()
os.environ.pop(bench_configs.history.ENV_VAR, None)
bench_configs.register()

import tyro  # noqa: E402

sys.path.insert(0, str(_UPSTREAM_SCRIPTS))

import compute_norm_stats as _upstream  # noqa: E402

if _upstream.__file__ != str(_UPSTREAM_SCRIPTS / "compute_norm_stats.py"):
    raise RuntimeError(f"resolved the wrong compute_norm_stats module: {_upstream.__file__}")

if __name__ == "__main__":
    tyro.cli(_upstream.main)
