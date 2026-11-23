from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import runpy
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types

_HERE = pathlib.Path(__file__).resolve().parent


def _load_sibling(name: str) -> types.ModuleType:
    module_name = f"groot_bench_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = _HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to locate {name} at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _run_dir(argv: list[str]) -> tuple[pathlib.Path, pathlib.Path]:
    parser = argparse.ArgumentParser(add_help=False)
    for flag in ("dataset-path", "output-dir", "experiment-name", "embodiment-tag", "modality-config-path"):
        parser.add_argument(f"--{flag}", f"--{flag.replace('-', '_')}", dest=flag.replace("-", "_"))
    known, _ = parser.parse_known_args(argv)
    if known.embodiment_tag is not None or known.modality_config_path is not None:
        raise SystemExit("finetune/train.py sets the embodiment itself; drop --embodiment-tag / --modality-config-path")
    if known.dataset_path is None or known.output_dir is None:
        raise SystemExit("--dataset-path and --output-dir are required")
    run_dir = pathlib.Path(known.output_dir)
    if known.experiment_name is not None:
        run_dir = run_dir / known.experiment_name
    return pathlib.Path(known.dataset_path), run_dir


def _schedule_args(argv: list[str]) -> tuple[int | None, float | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int)
    parser.add_argument("--warmup-ratio", "--warmup_ratio", dest="warmup_ratio", type=float)
    known, _ = parser.parse_known_args(argv)
    return known.max_steps, known.warmup_ratio


def main() -> None:
    history = _load_sibling("history")
    bench_embodiment = _load_sibling("bench_embodiment")
    parquet_images = _load_sibling("parquet_images")
    camera_vintage = _load_sibling("prep_dataset").load_camera_vintage()

    argv = sys.argv[1:]
    dataset_path, run_dir = _run_dir(argv)
    offsets = history.offsets_from_env()
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    schema = camera_vintage.validate_camera_stamp(
        info.get(camera_vintage.SCHEMA_INFO_KEY), source=str(dataset_path / "meta" / "info.json")
    )
    existing = tuple(sorted(p.name for p in run_dir.glob("checkpoint-*") if p.is_dir()))
    steps = tuple(int(name.removeprefix("checkpoint-")) for name in existing)
    camera_vintage.stamp_checkpoint_vintage(run_dir, schema, existing_steps=steps)
    history.stamp_run_dir(run_dir, offsets, existing_checkpoints=existing)
    restart_schedule = _load_sibling("restart_schedule")
    restart = restart_schedule.from_env()
    record = restart_schedule.schedule_record(*_schedule_args(argv), restart)
    restart_schedule.stamp_run_dir(run_dir, record, steps)
    backbone_attention = _load_sibling("backbone_attention")
    attention = backbone_attention.from_env()

    bench_embodiment.register(offsets)
    parquet_images.install()
    _load_sibling("hf_offline").install()
    if restart is not None:
        restart_schedule.install(restart)
        print(f"[bench] learning-rate restart {restart}", flush=True)
    if attention is not None:
        backbone_attention.install(attention, run_dir, steps)
        print(f"[bench] backbone attention: {attention} (BENCH_GROOT_ATTENTION)", flush=True)
    print(f"[bench] video keys {bench_embodiment.video_keys(offsets)}", flush=True)

    import gr00t.experiment.launch_finetune as launch_finetune

    sys.argv = [launch_finetune.__file__, *argv, "--embodiment-tag", bench_embodiment.EMBODIMENT_TAG_NAME]
    runpy.run_path(launch_finetune.__file__, run_name="__main__")


if __name__ == "__main__":
    main()
