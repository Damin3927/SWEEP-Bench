from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from sweep_bench.data.provenance import (
    COLLECTION_INFO_KEY,
    append_episode_provenance,
    collection_stamp_json,
)
from sweep_bench.data.vintage import INFO_KEY, schema_stamp

if TYPE_CHECKING:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    from sweep_bench.data.provenance import CollectionStamp, EpisodeProvenance
    from sweep_bench.data.schemas import BenchSchema


_FEATURE_PATCH_APPLIED = False


def _patch_hf_features_for_shape_1() -> None:
    global _FEATURE_PATCH_APPLIED
    if _FEATURE_PATCH_APPLIED:
        return

    import datasets
    from lerobot.common.datasets import lerobot_dataset as _lerobot_dataset
    from lerobot.common.datasets import utils as _lerobot_utils
    from lerobot.common.datasets.utils import DEFAULT_FEATURES

    original = _lerobot_utils.get_hf_features_from_features

    default_keys = set(DEFAULT_FEATURES.keys())

    def patched(features: dict) -> datasets.Features:
        hf = {}
        for key, ft in features.items():
            if ft["dtype"] == "video":
                continue
            if ft["dtype"] == "image":
                hf[key] = datasets.Image()
            elif ft["shape"] == (1,) and key not in default_keys:
                hf[key] = datasets.Sequence(length=1, feature=datasets.Value(dtype=ft["dtype"]))
            else:
                hf[key] = original({key: ft})[key]
        return datasets.Features(hf)

    vars(_lerobot_utils)["get_hf_features_from_features"] = patched
    vars(_lerobot_dataset)["get_hf_features_from_features"] = patched
    _FEATURE_PATCH_APPLIED = True


class LeRobotEpisodeRecorder:
    def __init__(
        self,
        repo_id: str,
        schema: BenchSchema,
        fps: int,
        root: Path | None = None,
        *,
        collection: CollectionStamp | None = None,
        overwrite: bool = False,
        image_writer_threads: int = 4,
        image_writer_processes: int = 0,
        use_videos: bool = True,
    ) -> None:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
        from lerobot.common.datasets.utils import write_info

        _patch_hf_features_for_shape_1()

        self.repo_id = repo_id
        self.schema = schema
        self.fps = fps
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        if self.root.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Dataset directory already exists: {self.root}. Pass overwrite=True or pick a new repo_id."
                )
            shutil.rmtree(self.root)

        self._dataset: LeRobotDataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=self.root,
            robot_type=schema.robot_type,
            features=schema.features,
            use_videos=use_videos,
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
        )

        self._dataset.meta.info[INFO_KEY] = schema_stamp(schema)
        if collection is not None:
            self._dataset.meta.info[COLLECTION_INFO_KEY] = collection_stamp_json(collection)
        write_info(self._dataset.meta.info, self.root)

    @property
    def num_episodes(self) -> int:
        return int(self._dataset.num_episodes)

    def add_frame(self, frame: dict) -> None:
        self._dataset.add_frame(frame)

    def save_episode(self, provenance: EpisodeProvenance | None = None) -> None:
        episode_index = self.num_episodes
        self._dataset.save_episode()
        if provenance is not None:
            append_episode_provenance(self.root, episode_index, provenance)

    def close(self) -> None:
        del self._dataset
