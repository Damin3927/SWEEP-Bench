from __future__ import annotations

import json
import os
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

ENV_VAR = "BENCH_ACTION_SPACE"
DATASET_ENV_VAR = "BENCH_EE_DATASET"
CHECKPOINT_STAMP = "bench_action_space.json"
METADATA_KEY = "action_space"
EULER_METADATA_KEY = "euler_convention"
EULER_MARKER = "meta/bench_euler_convention.json"

JOINT_POSITION = "joint_position"
EE_POSE = "ee_pose"
EE_POSE_DELTA = "ee_pose_delta"
ACTION_SPACES = (JOINT_POSITION, EE_POSE, EE_POSE_DELTA)
EE_ACTION_SPACES = (EE_POSE, EE_POSE_DELTA)
EULER_CONVENTIONS = ("rx_wrapped", "xz_wrapped")

POSE_DIM = 6
ACTION_DIM = 7
_STATE_KEY = "observation.state"
_ACTION_KEY = "action"


def from_env() -> str:
    raw = os.environ.get(ENV_VAR, "").strip() or JOINT_POSITION
    if raw not in ACTION_SPACES:
        raise ValueError(f"{ENV_VAR} must be one of {ACTION_SPACES} or empty, got {raw!r}")
    return raw


def read_euler_convention(root: pathlib.Path) -> str:
    path = pathlib.Path(root) / EULER_MARKER
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing, so the euler branch of the EE dataset {root} is unknown")
    convention = json.loads(path.read_text()).get(EULER_METADATA_KEY)
    if convention not in EULER_CONVENTIONS:
        raise ValueError(f"{path} names euler_convention={convention!r}; expected one of {EULER_CONVENTIONS}")
    return convention


def _column(table, key: str, width: int) -> np.ndarray:
    values = table.column(key).combine_chunks().flatten().to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(table.num_rows, width)


def load_ee_table(root: pathlib.Path, lengths: Sequence[int]) -> dict[str, np.ndarray]:
    root = pathlib.Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    if info["total_episodes"] < len(lengths):
        raise ValueError(f"{root} holds {info['total_episodes']} episodes, fewer than the {len(lengths)} to train on")
    columns = [_STATE_KEY, _ACTION_KEY, "episode_index", "frame_index", "index"]
    states, actions, episodes, frames, indices = [], [], [], [], []
    for episode, length in enumerate(lengths):
        path = root / info["data_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode)
        table = pq.read_table(path, columns=columns)
        if table.num_rows != length:
            raise ValueError(f"{path} has {table.num_rows} frames, but the DROID episode {episode} has {length}")
        states.append(_column(table, _STATE_KEY, POSE_DIM))
        actions.append(_column(table, _ACTION_KEY, ACTION_DIM))
        episodes.append(table.column("episode_index").to_numpy())
        frames.append(table.column("frame_index").to_numpy())
        indices.append(table.column("index").to_numpy())
    counts = np.asarray(lengths, dtype=np.int64)
    episode_end = np.cumsum(counts)
    result = {
        "state": np.concatenate(states),
        "action": np.concatenate(actions),
        "episode_index": np.concatenate(episodes).astype(np.int64),
        "frame_index": np.concatenate(frames).astype(np.int64),
        "episode_end": episode_end,
    }
    if not np.array_equal(result["episode_index"], np.repeat(np.arange(len(counts)), counts)):
        raise ValueError(f"{root}: episode_index is not each file's own episode")
    if not np.array_equal(result["frame_index"], np.arange(episode_end[-1]) - np.repeat(episode_end - counts, counts)):
        raise ValueError(f"{root}: frame_index does not count 0, 1, ... within each episode")
    if not np.array_equal(np.concatenate(indices), np.arange(episode_end[-1])):
        raise ValueError(f"{root}: index is not contiguous from 0, so it cannot be looked up by the DROID index")
    if not (np.isfinite(result["state"]).all() and np.isfinite(result["action"]).all()):
        raise ValueError(f"{root}: non-finite EE state or action")
    return result


def lookup_ee(
    table: dict[str, np.ndarray], *, index: int, episode: int, frame: int, horizon: int, gripper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= index < len(table["state"]) or (table["episode_index"][index], table["frame_index"][index]) != (
        episode,
        frame,
    ):
        raise ValueError(
            f"EE row {index} is not episode {episode} frame {frame}: the EE and DROID datasets are not aligned"
        )
    rows = np.minimum(index + np.arange(horizon), table["episode_end"][episode] - 1)
    actions = table["action"][rows]
    if not np.array_equal(actions[:, -1], np.asarray(gripper, dtype=np.float32)):
        raise ValueError(
            f"episode {episode} frame {frame}: the EE gripper chunk {actions[:, -1]} differs from the DROID one "
            f"{np.asarray(gripper)}; the EE and DROID datasets are not aligned"
        )
    return table["state"][index].copy(), actions


def pose_delta(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    result = np.array(actions, dtype=np.float32)
    result[..., :POSE_DIM] -= np.expand_dims(np.asarray(state)[..., :POSE_DIM], axis=-2)
    result[..., 3:POSE_DIM] = np.mod(result[..., 3:POSE_DIM] + np.pi, 2 * np.pi) - np.pi
    return result


def _stamp_record(space: str, ee_dataset: str, euler: str) -> dict:
    return {METADATA_KEY: space, "ee_dataset": ee_dataset, EULER_METADATA_KEY: euler}


def stamp_run_dir(
    run_dir: pathlib.Path, space: str, ee_dataset: str | None, euler: str | None, *, existing_steps: tuple[int, ...]
) -> None:
    path = pathlib.Path(run_dir) / CHECKPOINT_STAMP
    recorded = read_stamp(path) if path.is_file() else None
    if space == JOINT_POSITION:
        if recorded is not None:
            raise RuntimeError(
                f"{run_dir} holds {recorded[METADATA_KEY]} checkpoints, but {ENV_VAR} is {JOINT_POSITION}. "
                f"Use a fresh --exp-name."
            )
        return
    if not ee_dataset or euler is None:
        raise RuntimeError(f"{ENV_VAR}={space} needs {DATASET_ENV_VAR} and its euler convention")
    record = _stamp_record(space, ee_dataset, euler)
    if recorded is not None:
        if recorded != record:
            raise RuntimeError(
                f"{run_dir} was trained with {recorded}, but this run uses {record}. Use a fresh --exp-name."
            )
        return
    if existing_steps:
        raise RuntimeError(
            f"{run_dir} already holds checkpoints {existing_steps} without {CHECKPOINT_STAMP}, so they were trained "
            f"on joint positions; resuming them on {space} would mix the two. Use a fresh --exp-name."
        )
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(path)
    print(f"[bench] stamped {path} with {record}", flush=True)


def read_stamp(path: pathlib.Path) -> dict:
    data = json.loads(pathlib.Path(path).read_text())
    if (
        not isinstance(data, dict)
        or data.get(METADATA_KEY) not in EE_ACTION_SPACES
        or not data.get("ee_dataset")
        or data.get(EULER_METADATA_KEY) not in EULER_CONVENTIONS
    ):
        raise ValueError(f"{path} is not a valid {CHECKPOINT_STAMP}: {data!r}")
    return data


def checkpoint_stamp(ckpt_dir: str | pathlib.Path, *, download: Callable[[str], str] | None = None) -> dict | None:
    if "://" in str(ckpt_dir):
        return _remote_checkpoint_stamp(str(ckpt_dir), download)
    path = pathlib.Path(ckpt_dir)
    for candidate in (path, *path.parents):
        stamp = candidate / CHECKPOINT_STAMP
        if stamp.is_file():
            return read_stamp(stamp)
    return None


def _remote_checkpoint_stamp(uri: str, download: Callable[[str], str] | None) -> dict | None:
    if download is None:
        from openpi.shared.download import maybe_download

        download = maybe_download
    scheme, _, path = uri.partition("://")
    segments = [segment for segment in path.split("/") if segment]
    for depth in range(len(segments), 0, -1):
        candidate = f"{scheme}://{'/'.join(segments[:depth])}/{CHECKPOINT_STAMP}"
        try:
            local = download(candidate)
        except FileNotFoundError:
            continue
        except Exception as exc:
            raise RuntimeError(f"could not read {candidate} ({exc}), so {uri}'s action space is unknown") from exc
        return read_stamp(pathlib.Path(local))
    return None


def check_serve(
    ckpt_dir: str | pathlib.Path, configured: str, *, download: Callable[[str], str] | None = None
) -> dict[str, str]:
    recorded = checkpoint_stamp(ckpt_dir, download=download)
    trained = JOINT_POSITION if recorded is None else recorded[METADATA_KEY]
    if trained != configured:
        raise RuntimeError(
            f"checkpoint {ckpt_dir} was trained on {trained} (per {CHECKPOINT_STAMP}"
            f"{'' if recorded else ' being absent'}), but {ENV_VAR} configures {configured}. "
            f"Serve it with {ENV_VAR}={trained}."
        )
    if recorded is None:
        return {METADATA_KEY: JOINT_POSITION}
    return {METADATA_KEY: trained, EULER_METADATA_KEY: recorded[EULER_METADATA_KEY]}
