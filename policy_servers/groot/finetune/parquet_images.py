from __future__ import annotations

import functools
import importlib.util
import io
import pathlib
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import os
    import types

    import pandas as pd
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.types import ModalityConfig


def _load_history() -> types.ModuleType:
    name = "groot_bench_history"
    if name in sys.modules:
        return sys.modules[name]
    path = pathlib.Path(__file__).resolve().parent / "history.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to locate history at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


history = _load_history()

IMAGE_COLUMNS = (history.EXTERIOR_COLUMN, "wrist_image_left")


def source_of(key: str) -> tuple[str, int]:
    parsed = history.parse_video_key(key)
    if parsed is not None:
        return parsed
    if key not in IMAGE_COLUMNS:
        raise KeyError(f"video key {key!r} is neither an image column {IMAGE_COLUMNS} nor a history key")
    return key, 0


ImageCell = dict[str, bytes | str | None] | bytes


def decode_png(cell: ImageCell) -> np.ndarray:
    from PIL import Image

    data = cell.get("bytes") if isinstance(cell, dict) else cell
    if not isinstance(data, bytes):
        raise TypeError(f"expected PNG bytes in an image cell, got {type(data).__name__}")
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"))


def select_frames(frames: list[np.ndarray], indices: np.ndarray, offset: int) -> np.ndarray:
    return np.stack([frames[max(0, int(i) + offset)] for i in indices])


class _ParquetWithoutImages:
    def __getattr__(self, name: str) -> object:
        import pandas

        return getattr(pandas, name)

    @staticmethod
    def read_parquet(path: str | os.PathLike[str], columns: list[str] | None = None) -> pd.DataFrame:
        import pandas
        import pyarrow.parquet as pq

        if columns is None:
            columns = [c for c in pq.read_schema(path).names if c not in IMAGE_COLUMNS]
        return pandas.read_parquet(path, columns=columns)


def _episode_parquet(loader: LeRobotEpisodeLoader, episode_index: int) -> pathlib.Path:
    chunk = episode_index // loader.chunk_size
    return loader.dataset_path / loader.data_path_pattern.format(episode_chunk=chunk, episode_index=episode_index)


def _load_video_data(self: LeRobotEpisodeLoader, episode_index: int, indices: np.ndarray) -> dict[str, np.ndarray]:
    import pandas

    if "video" not in self.modality_configs:
        return {}
    keys = self.modality_configs["video"].modality_keys
    sources = {key: source_of(key) for key in keys}
    columns = sorted({column for column, _ in sources.values()})
    table = pandas.read_parquet(_episode_parquet(self, episode_index), columns=columns)
    decoded = {column: [decode_png(cell) for cell in table[column]] for column in columns}
    return {key: select_frames(decoded[column], indices, offset) for key, (column, offset) in sources.items()}


def install() -> None:
    from gr00t.data import stats
    from gr00t.data.dataset import lerobot_episode_loader

    loader_cls = lerobot_episode_loader.LeRobotEpisodeLoader
    if getattr(loader_cls, "_bench_parquet_images", False):
        return
    original_parse = loader_cls._parse_and_validate_modality_configs

    @functools.wraps(original_parse)
    def parse(self: LeRobotEpisodeLoader, modality_configs: dict[str, ModalityConfig]) -> dict[str, ModalityConfig]:
        video = modality_configs.get("video")
        if video is not None and "video" in self.modality_meta:
            for key in video.modality_keys:
                column, offset = source_of(key)
                if offset:
                    self.modality_meta["video"][key] = {"original_key": column}
        return original_parse(self, modality_configs)

    setattr(loader_cls, "_parse_and_validate_modality_configs", parse)  # noqa: B010
    setattr(loader_cls, "_load_video_data", _load_video_data)  # noqa: B010
    setattr(lerobot_episode_loader, "pd", _ParquetWithoutImages())  # noqa: B010
    setattr(stats, "pd", _ParquetWithoutImages())  # noqa: B010
    setattr(loader_cls, "_bench_parquet_images", True)  # noqa: B010
