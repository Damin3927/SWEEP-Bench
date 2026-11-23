"""Build the GR00T view of a bench LeRobot dataset and its normalization statistics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import shutil
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types


def _load_sibling(name: str) -> types.ModuleType:
    path = pathlib.Path(__file__).resolve().parent / f"{name}.py"
    module_name = f"groot_bench_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to locate {name} at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_camera_vintage() -> types.ModuleType:
    path = pathlib.Path(__file__).resolve().parents[2] / "openpi" / "finetune" / "camera_vintage.py"
    name = "groot_bench_camera_vintage"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to locate camera_vintage at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_COPIED_META = ("info.json", "tasks.jsonl")


def build_overlay(source: pathlib.Path, out: pathlib.Path, max_episodes: int | None) -> int:
    bench_embodiment = _load_sibling("bench_embodiment")
    info = json.loads((source / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"{source} is LeRobot {info.get('codebase_version')!r}; GR00T reads v2.x datasets")
    episodes = (source / "meta" / "episodes.jsonl").read_text().splitlines()
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for name in _COPIED_META:
        shutil.copyfile(source / "meta" / name, meta / name)
    (meta / "episodes.jsonl").write_text("\n".join(episodes) + "\n")
    (meta / "modality.json").write_text(json.dumps(bench_embodiment.modality_json(), indent=2))
    data = out / "data"
    if data.exists():
        shutil.rmtree(data)
    for line in episodes:
        index = json.loads(line)["episode_index"]
        rel = info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
        target = source / rel
        if not target.is_file():
            raise FileNotFoundError(target)
        link = out / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
    return len(episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    out = args.out.expanduser().resolve()

    info = json.loads((source / "meta" / "info.json").read_text())
    vintage = load_camera_vintage()
    vintage.validate_camera_stamp(info.get(vintage.SCHEMA_INFO_KEY), source=str(source / "meta" / "info.json"))

    count = build_overlay(source, out, args.max_episodes)
    print(f"[bench] overlay {out}: {count} episodes from {source}", flush=True)

    bench_embodiment = _load_sibling("bench_embodiment")
    parquet_images = _load_sibling("parquet_images")
    parquet_images.install()
    bench_embodiment.register(())
    from gr00t.data import stats

    stats.generate_stats(out)
    stats.generate_rel_stats(out, bench_embodiment.embodiment_tag())
    print(f"[bench] wrote {out / 'meta' / 'stats.json'} and {out / 'meta' / 'relative_stats.json'}", flush=True)


if __name__ == "__main__":
    main()
