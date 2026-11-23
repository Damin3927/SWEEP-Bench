"""Convert a bench pi05 JAX checkpoint to the ``model.safetensors`` that ``serve.py --fast=graph`` reads."""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str, path: pathlib.Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to locate {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stamp_names(bench_configs: types.ModuleType) -> tuple[str, ...]:
    return tuple(
        m.CHECKPOINT_STAMP
        for m in (
            bench_configs.camera_vintage,
            bench_configs.history,
            bench_configs.gt_speed,
            bench_configs.action_space,
            bench_configs.episode_sampling,
        )
    )


def copy_stamps(source: pathlib.Path, output_dir: pathlib.Path, names: tuple[str, ...]) -> list[str]:
    copied = []
    for name in names:
        found = next((d / name for d in (source, *source.parents) if (d / name).is_file()), None)
        if found is not None:
            shutil.copy2(found, output_dir / name)
            copied.append(name)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True, help="JAX step dir, e.g. <run>/29999 (holds params/)")
    parser.add_argument("--output-dir", required=True, help="new run dir; the step dir is created inside it")
    args = parser.parse_args()
    source = pathlib.Path(args.checkpoint_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    out_step = output_dir / source.name
    if out_step.exists():
        raise FileExistsError(f"{out_step} exists; pick a fresh --output-dir")

    import openpi.training.checkpoints  # noqa: F401

    bench_configs = _load("openpi_bench_configs", _HERE / "bench_configs.py")
    bench_configs.register()
    from openpi.training import config as _config

    train_config = _config.get_config(bench_configs.BENCH_CONFIG_NAME)
    model = train_config.model
    if not getattr(model, "pi05", False):
        raise ValueError(f"only pi05 checkpoints convert to the fast path, got {type(model).__name__}")
    bench_configs.check_checkpoint_history(source)
    bench_configs.check_checkpoint_gt_speed(source)
    bench_configs.check_checkpoint_action_space(source)

    convert = _load("openpi_convert_jax_to_pytorch", pathlib.Path("examples/convert_jax_model_to_pytorch.py").resolve())
    with tempfile.TemporaryDirectory() as tmp:
        link = pathlib.Path(tmp) / "pi05" / source.name
        link.parent.mkdir()
        link.symlink_to(source, target_is_directory=True)
        convert.convert_pi0_checkpoint(str(link), "bfloat16", str(out_step), model)
    shutil.copytree(source / "assets", out_step / "assets", dirs_exist_ok=True)
    copied = copy_stamps(source, output_dir, stamp_names(bench_configs))
    print(f"converted {source} -> {out_step}; stamps {copied}")


if __name__ == "__main__":
    main()
