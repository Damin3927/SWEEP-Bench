from __future__ import annotations

import dataclasses
import functools
import importlib.util
import json
import logging
import os
import pathlib
import re
from typing import TYPE_CHECKING

from typing_extensions import override

import flax.traverse_util
import jax
import numpy as np
import optax
from flax import nnx
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets import lerobot_dataset

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.shared.array_typing as at
import openpi.training.config as _c
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from openpi.policies import droid_policy, libero_policy

if TYPE_CHECKING:
    import types

BENCH_CONFIG_NAME = "bench"
DEFAULT_NUM_STEPS = {"pi05_base": 30_000, "pi0_fast_base": 130_000}
SCHEDULE_STAMP = "bench_lr_schedule.json"
SAVE_CONCURRENT_GB = 6


def _load_sibling(name: str) -> types.ModuleType:
    path = pathlib.Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"openpi_bench_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to locate {name} at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


camera_vintage = _load_sibling("camera_vintage")
history = _load_sibling("history")
gt_speed = _load_sibling("gt_speed")
episode_sampling = _load_sibling("episode_sampling")
action_space = _load_sibling("action_space")

POLICY_PARAMS = {
    "pi05_base": "gs://openpi-assets/checkpoints/pi05_base/params",
    "pi0_fast_base": "gs://openpi-assets/checkpoints/pi0_fast_base/params",
}


@dataclasses.dataclass(frozen=True)
class BenchDataConfig(_c.DataConfig):
    exterior_history: tuple[int, ...] = ()
    max_episodes: int | None = None
    gt_speed: bool = False
    gt_speed_list: str | None = None
    episode_uniform: bool = False
    bench_action_space: str = action_space.JOINT_POSITION
    ee_dataset: str | None = None


@dataclasses.dataclass(frozen=True)
class BenchDroidInputs(_transforms.DataTransformFn):
    model_type: _model.ModelType
    exterior_history: tuple[int, ...] = ()
    gt_speed: bool = False
    ee_pose: bool = False

    def __call__(self, data: dict) -> dict:
        if self.ee_pose:
            data = {**data, "observation/joint_position": data["observation/ee_pose"]}
        inputs = self._droid_inputs(data)
        if self.gt_speed != (gt_speed.REQUEST_KEY in data):
            raise ValueError(
                f"this model {'reads' if self.gt_speed else 'does not read'} GT speed, but the sample "
                f"{'lacks' if self.gt_speed else 'carries'} {gt_speed.REQUEST_KEY!r}"
            )
        if self.gt_speed:
            speed = gt_speed.check_speed(float(np.asarray(data[gt_speed.REQUEST_KEY])), gt_speed.REQUEST_KEY)
            inputs[gt_speed.REQUEST_KEY] = np.float32(speed)
        return inputs

    def _droid_inputs(self, data: dict) -> dict:
        droid = droid_policy.DroidInputs(model_type=self.model_type)
        if not self.exterior_history:
            return _drop_empty_slot(droid(data), self.model_type)
        if self.model_type not in (_model.ModelType.PI05, _model.ModelType.PI0_FAST):
            raise ValueError(f"exterior history is implemented for pi05 and pi0-FAST only, got {self.model_type}")
        frames = np.asarray(data["observation/exterior_image_1_left"])
        expected = len(self.exterior_history) + 1
        if frames.ndim != 4 or frames.shape[0] != expected:
            raise ValueError(
                f"exterior history {self.exterior_history} needs observation/exterior_image_1_left stacked as "
                f"({expected}, H, W, C) or ({expected}, C, H, W), got shape {frames.shape}"
            )
        inputs = _drop_empty_slot(droid({**data, "observation/exterior_image_1_left": frames[-1]}), self.model_type)
        for offset, frame in zip(self.exterior_history, frames[:-1], strict=True):
            key = history.image_key(offset)
            inputs["image"][key] = droid_policy._parse_image(frame)
            inputs["image_mask"][key] = np.True_
        return inputs


@dataclasses.dataclass(frozen=True)
class EpisodeBeltSpeed(_transforms.DataTransformFn):
    speeds: tuple[float, ...]

    def __call__(self, data: dict) -> dict:
        return {**data, gt_speed.REQUEST_KEY: np.float32(self.speeds[int(data["episode_index"])])}


@dataclasses.dataclass(frozen=True)
class EePoseLookup(_transforms.DataTransformFn):
    table: dict[str, np.ndarray]
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        ee_pose, actions = action_space.lookup_ee(
            self.table,
            index=int(data["index"]),
            episode=int(data["episode_index"]),
            frame=int(data["frame_index"]),
            horizon=self.action_horizon,
            gripper=np.asarray(data["actions"])[:, -1],
        )
        return {**data, "ee_pose": ee_pose, "actions": actions}


@dataclasses.dataclass(frozen=True)
class EePoseDeltaActions(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        return {**data, "actions": action_space.pose_delta(data["actions"], data["state"])}


@dataclasses.dataclass(frozen=True)
class GtSpeedToStateSlot(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        data = dict(data)
        speed = float(np.asarray(data.pop(gt_speed.REQUEST_KEY)))
        state = np.array(data["state"], dtype=np.float32)
        if np.any(state[..., gt_speed.STATE_SLOT] != 0):
            raise ValueError(f"state slot {gt_speed.STATE_SLOT} is not padding: {state}")
        state[..., gt_speed.STATE_SLOT] = gt_speed.encode(speed)
        data["state"] = state
        return data


def _drop_empty_slot(inputs: dict, model_type: _model.ModelType) -> dict:
    if model_type == _model.ModelType.PI05:
        del inputs["image"]["right_wrist_0_rgb"], inputs["image_mask"]["right_wrist_0_rgb"]
    return inputs


def history_image_keys(exterior_history: tuple[int, ...]) -> tuple[str, ...]:
    return ("base_0_rgb", "left_wrist_0_rgb", *(history.image_key(o) for o in exterior_history))


def install_history_preprocess(exterior_history: tuple[int, ...] | None) -> None:
    if _pi0._model is not _model:
        raise RuntimeError("openpi.models.pi0 no longer calls preprocess_observation through openpi.models.model")
    original = getattr(_model.preprocess_observation, "__wrapped__", _model.preprocess_observation)
    if exterior_history is None:
        _model.preprocess_observation = original
        return
    keys = history_image_keys(exterior_history)

    @functools.wraps(original)
    def patched(rng, observation, *, train=False, image_keys=_model.IMAGE_KEYS, **kwargs):
        if tuple(image_keys) == _model.IMAGE_KEYS:
            image_keys = keys
        return original(rng, observation, train=train, image_keys=image_keys, **kwargs)

    _model.preprocess_observation = patched


_ORIGINAL_CREATE_TORCH_DATASET = _data_loader.create_torch_dataset


def _create_torch_dataset(data_config: _c.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig):
    if not isinstance(data_config, BenchDataConfig) or (
        not data_config.exterior_history
        and data_config.max_episodes is None
        and not data_config.gt_speed
        and data_config.bench_action_space == action_space.JOINT_POSITION
    ):
        return _ORIGINAL_CREATE_TORCH_DATASET(data_config, action_horizon, model_config)
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    delta_timestamps = {key: [t / meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys}
    if data_config.exterior_history:
        delta_timestamps["exterior_image_1_left"] = [o / meta.fps for o in (*data_config.exterior_history, 0)]
    episodes = None if data_config.max_episodes is None else list(range(data_config.max_episodes))
    dataset = lerobot_dataset.LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, episodes=episodes)
    transforms = []
    if data_config.prompt_from_task:
        transforms.append(_transforms.PromptFromLeRobotTask(meta.tasks))
    if data_config.gt_speed:
        if not data_config.gt_speed_list:
            raise ValueError(f"{gt_speed.ENV_VAR}=1 needs {gt_speed.LIST_ENV_VAR}, the per-episode speed list")
        transforms.append(
            EpisodeBeltSpeed(gt_speed.load_episode_speeds(data_config.gt_speed_list, meta.total_episodes))
        )
    if data_config.bench_action_space != action_space.JOINT_POSITION:
        if not data_config.ee_dataset:
            raise ValueError(
                f"{action_space.ENV_VAR}={data_config.bench_action_space} needs {action_space.DATASET_ENV_VAR}, "
                f"the EE-DOM conversion of {repo_id}"
            )
        count = meta.total_episodes if data_config.max_episodes is None else data_config.max_episodes
        lengths = [meta.episodes[i]["length"] for i in range(count)]
        table = action_space.load_ee_table(HF_LEROBOT_HOME / data_config.ee_dataset, lengths)
        transforms.append(EePoseLookup(table, action_horizon))
    return _data_loader.TransformedDataset(dataset, transforms) if transforms else dataset


_ORIGINAL_CREATE_TORCH_DATA_LOADER = _data_loader.create_torch_data_loader


def _create_torch_data_loader(data_config: _c.DataConfig, *args, seed: int = 0, **kwargs):
    if not isinstance(data_config, BenchDataConfig) or not data_config.episode_uniform:
        return _ORIGINAL_CREATE_TORCH_DATA_LOADER(data_config, *args, seed=seed, **kwargs)
    episodes = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id).episodes
    count = len(episodes) if data_config.max_episodes is None else data_config.max_episodes
    lengths = [episodes[i]["length"] for i in range(count)]
    loader_cls = _data_loader.TorchDataLoader

    def episode_uniform_loader(dataset, local_batch_size, *, sampler=None, **loader_kwargs):
        if sampler is not None:
            raise NotImplementedError(f"episode-uniform sampling does not compose with {type(sampler).__name__}")
        sampler = episode_sampling.sampler(lengths, len(dataset), seed)
        return loader_cls(dataset, local_batch_size, sampler=sampler, **loader_kwargs)

    _data_loader.TorchDataLoader = episode_uniform_loader
    try:
        return _ORIGINAL_CREATE_TORCH_DATA_LOADER(data_config, *args, seed=seed, **kwargs)
    finally:
        _data_loader.TorchDataLoader = loader_cls


@dataclasses.dataclass(frozen=True)
class PromptTokensFit(_transforms.DataTransformFn):
    max_token_len: int

    def __call__(self, data: dict) -> dict:
        used = int(np.asarray(data["tokenized_prompt_mask"]).sum())
        if used >= self.max_token_len:
            raise ValueError(
                f"the prompt and state fill all {self.max_token_len} pi05 prompt tokens and may have been "
                "truncated; raise PI05_MAX_TOKEN_LEN in bench_configs.py"
            )
        return data


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDPositionDataConfig(_c.DataConfigFactory):
    @override
    def create(self, assets_dirs, model_config):
        base = self.base_config
        speed = isinstance(base, BenchDataConfig) and base.gt_speed
        space = base.bench_action_space if isinstance(base, BenchDataConfig) else action_space.JOINT_POSITION
        ee = space != action_space.JOINT_POSITION
        state_key = "ee_pose" if ee else "joint_position"
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        f"observation/{state_key}": state_key,
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                        **({gt_speed.REQUEST_KEY: gt_speed.REQUEST_KEY} if speed else {}),
                    }
                )
            ]
        )
        exterior_history = base.exterior_history if isinstance(base, BenchDataConfig) else ()
        data_transforms = _transforms.Group(
            inputs=[
                BenchDroidInputs(
                    model_type=model_config.model_type, exterior_history=exterior_history, gt_speed=speed, ee_pose=ee
                )
            ],
            outputs=[libero_policy.LiberoOutputs() if ee else droid_policy.DroidOutputs()],
        )
        if space == action_space.JOINT_POSITION:
            delta_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_mask)],
                outputs=[_transforms.AbsoluteActions(delta_mask)],
            )
        elif space == action_space.EE_POSE_DELTA:
            data_transforms = data_transforms.push(
                inputs=[EePoseDeltaActions()],
                outputs=[_transforms.AbsoluteActions(_transforms.make_bool_mask(action_space.POSE_DIM, -1))],
            )
        model_transforms = _c.ModelTransformFactory()(model_config)
        if model_config.model_type == _model.ModelType.PI05:
            model_transforms = model_transforms.push(inputs=[PromptTokensFit(model_config.max_token_len)])
        if speed:
            model_transforms = model_transforms.push(inputs=[GtSpeedToStateSlot()])
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


def _data(
    repo_id: str,
    *,
    exterior_history: tuple[int, ...],
    max_episodes: int | None,
    gt_speed_input: bool = False,
    gt_speed_list: str | None = None,
    episode_uniform: bool = False,
    space: str = action_space.JOINT_POSITION,
    ee_dataset: str | None = None,
) -> LeRobotDROIDPositionDataConfig:
    return LeRobotDROIDPositionDataConfig(
        repo_id=repo_id,
        base_config=BenchDataConfig(
            prompt_from_task=True,
            exterior_history=exterior_history,
            max_episodes=max_episodes,
            gt_speed=gt_speed_input,
            gt_speed_list=gt_speed_list,
            episode_uniform=episode_uniform,
            bench_action_space=space,
            ee_dataset=ee_dataset,
        ),
    )


PI05_MAX_TOKEN_LEN = 128


class GtSpeedPi0(_pi0.Pi0):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        width = self.action_in_proj.out_features
        self.speed_mlp_in = nnx.Linear(width, width, rngs=rngs)
        self.speed_mlp_out = nnx.Linear(width, width, kernel_init=nnx.initializers.zeros, rngs=rngs)

    @override
    def embed_suffix(self, obs: _model.Observation, noisy_actions: _model.Actions, timestep):
        tokens, input_mask, ar_mask, adarms_cond = super().embed_suffix(obs, noisy_actions, timestep)
        speed_emb = _pi0.posemb_sincos(
            obs.state[:, gt_speed.STATE_SLOT], self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )
        speed_emb = self.speed_mlp_out(nnx.swish(self.speed_mlp_in(speed_emb)))
        return tokens, input_mask, ar_mask, adarms_cond + speed_emb


class GtSpeedPi0Config(pi0_config.Pi0Config):
    @override
    def create(self, rng) -> GtSpeedPi0:
        if not self.pi05:
            raise ValueError("GT speed is implemented for pi05 only")
        return GtSpeedPi0(self, rngs=nnx.Rngs(rng))


_GT_SPEED_PARAMS = re.compile(r"speed_mlp_(in|out)/.*")


def add_missing_speed_params(loaded: dict, reference: dict) -> dict:
    flat = flax.traverse_util.flatten_dict(loaded, sep="/")
    for key, value in flax.traverse_util.flatten_dict(reference, sep="/").items():
        if _GT_SPEED_PARAMS.fullmatch(key):
            flat.setdefault(key, value)
    return flax.traverse_util.unflatten_dict(flat, sep="/")


class GtSpeedWeightLoader(weight_loaders.CheckpointWeightLoader):
    @override
    def load(self, params):
        return add_missing_speed_params(super().load(params), params)


def _pi05(*, gt_speed_input: bool = False) -> pi0_config.Pi0Config:
    config_cls = GtSpeedPi0Config if gt_speed_input else pi0_config.Pi0Config
    return config_cls(pi05=True, action_dim=32, action_horizon=16, max_token_len=PI05_MAX_TOKEN_LEN)


PI0_FAST_MAX_TOKEN_LEN = 180
FAST_DECODER_KEY = "fast_decoder"


class GeneratedIdsFASTTokenizer(_tokenizer.FASTTokenizer):
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        super().__init__(max_len, fast_tokenizer_path)
        pg = self._paligemma_tokenizer
        self._action_marker = pg.encode("Action: ")
        self._action_end = {pg.piece_to_id("|"), pg.eos_id(), pg.pad_id()}

    @override
    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        ids = tokens.tolist()
        n = len(self._action_marker)
        start = next((i + n for i in range(len(ids) - n + 1) if ids[i : i + n] == self._action_marker), None)
        if start is None:
            logging.warning("No 'Action: ' marker in the generated ids, returning the zero chunk: %s", ids)
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        end = next((i for i in range(start, len(ids)) if ids[i] in self._action_end), len(ids))
        fast_ids = self._act_tokens_to_paligemma_tokens(ids[start:end]).tolist()
        return self._fast_tokenizer.decode([fast_ids], time_horizon=action_horizon, action_dim=action_dim)[0]


def fast_decoder(model: pi0_fast.Pi0FASTConfig) -> str:
    return "generated_ids" if model.fast_model_tokenizer is GeneratedIdsFASTTokenizer else "stock_text_round_trip"


def _pi0_fast() -> pi0_fast.Pi0FASTConfig:
    return pi0_fast.Pi0FASTConfig(
        action_dim=8,
        action_horizon=16,
        max_token_len=PI0_FAST_MAX_TOKEN_LEN,
        paligemma_variant="gemma_2b",
        fast_model_tokenizer=GeneratedIdsFASTTokenizer,
    )


_MODELS = {"pi05_base": _pi05, "pi0_fast_base": _pi0_fast}


@dataclasses.dataclass(frozen=True)
class RestartedCosineSchedule(_optimizer.LRScheduleConfig):
    cycle: _optimizer.CosineDecaySchedule
    num_steps: int

    def create(self) -> optax.Schedule:
        again = optax.warmup_cosine_decay_schedule(
            init_value=self.cycle.decay_lr,
            peak_value=self.cycle.peak_lr,
            warmup_steps=self.cycle.warmup_steps,
            decay_steps=self.num_steps - self.cycle.decay_steps,
            end_value=self.cycle.decay_lr,
        )
        return optax.join_schedules([self.cycle.create(), again], [self.cycle.decay_steps])


def build_config(
    *,
    repo_id: str,
    num_steps: int,
    policy: str,
    fsdp_devices: int = 1,
    exterior_history: tuple[int, ...] = (),
    num_workers: int = 2,
    keep_period: int | None = 1000,
    max_episodes: int | None = None,
    gt_speed_input: bool = False,
    gt_speed_list: str | None = None,
    episode_uniform: bool = False,
    space: str = action_space.JOINT_POSITION,
    ee_dataset: str | None = None,
) -> _c.TrainConfig:
    if policy not in POLICY_PARAMS:
        raise ValueError(f"BENCH_POLICY must be one of {sorted(POLICY_PARAMS)}, got {policy!r}")
    if fsdp_devices < 1:
        raise ValueError(f"BENCH_FSDP_DEVICES must be >= 1, got {fsdp_devices!r}")
    if gt_speed_input and policy != "pi05_base":
        raise ValueError(f"{gt_speed.ENV_VAR}=1 is implemented for pi05_base only, got {policy!r}")
    if space not in action_space.ACTION_SPACES:
        raise ValueError(f"{action_space.ENV_VAR} must be one of {action_space.ACTION_SPACES}, got {space!r}")
    if space != action_space.JOINT_POSITION and policy != "pi05_base":
        raise ValueError(f"{action_space.ENV_VAR}={space} is implemented for pi05_base only, got {policy!r}")
    if space != action_space.JOINT_POSITION and gt_speed_input:
        raise ValueError(f"{gt_speed.ENV_VAR}=1 is implemented for {action_space.JOINT_POSITION} only, got {space!r}")
    cycle = _optimizer.CosineDecaySchedule()
    restart = policy == "pi0_fast_base" and num_steps > cycle.decay_steps
    if restart and num_steps <= cycle.decay_steps + cycle.warmup_steps:
        raise ValueError(
            f"{policy} restarts its cosine at step {cycle.decay_steps}: BENCH_NUM_STEPS must be at most that or "
            f"more than {cycle.decay_steps + cycle.warmup_steps}, got {num_steps}"
        )

    exterior_history = history.validate_offsets(exterior_history)

    model = _pi05(gt_speed_input=True) if gt_speed_input else _MODELS[policy]()
    data = _data(
        repo_id,
        exterior_history=exterior_history,
        max_episodes=max_episodes,
        gt_speed_input=gt_speed_input,
        gt_speed_list=gt_speed_list,
        episode_uniform=episode_uniform,
        space=space,
        ee_dataset=ee_dataset,
    )
    loader_cls = GtSpeedWeightLoader if gt_speed_input else weight_loaders.CheckpointWeightLoader
    weight_loader = loader_cls(POLICY_PARAMS[policy])
    common = {
        "name": BENCH_CONFIG_NAME,
        "model": model,
        "data": data,
        "weight_loader": weight_loader,
        "num_train_steps": num_steps,
        "keep_period": keep_period,
        "fsdp_devices": fsdp_devices,
        "num_workers": num_workers,
        **({} if space == action_space.JOINT_POSITION else {"assets_base_dir": f"./assets/{space}"}),
        "policy_metadata": {
            history.METADATA_KEY: list(exterior_history),
            gt_speed.METADATA_KEY: gt_speed_input,
            **({FAST_DECODER_KEY: fast_decoder(model)} if model.model_type == _model.ModelType.PI0_FAST else {}),
        },
    }
    if policy == "pi05_base":
        libero = _c.get_config("pi05_libero")
        common.update(
            batch_size=libero.batch_size,
            lr_schedule=libero.lr_schedule,
            optimizer=libero.optimizer,
            ema_decay=libero.ema_decay,
        )
    else:
        common["batch_size"] = 32
    if restart:
        common["lr_schedule"] = RestartedCosineSchedule(cycle=cycle, num_steps=num_steps)
    return _c.TrainConfig(**common)


def register() -> None:
    repo_id = os.environ.get("BENCH_DATASET_REPO")
    if not repo_id:
        return
    exterior_history = history.offsets_from_env()
    keep_period = int(os.environ.get("BENCH_KEEP_PERIOD") or 1000)
    max_episodes = os.environ.get("BENCH_MAX_EPISODES")
    policy = os.environ.get("BENCH_POLICY") or "pi05_base"
    cfg = build_config(
        repo_id=repo_id,
        num_steps=int(os.environ.get("BENCH_NUM_STEPS") or DEFAULT_NUM_STEPS.get(policy, 0)),
        policy=policy,
        fsdp_devices=int(os.environ.get("BENCH_FSDP_DEVICES") or 1),
        exterior_history=exterior_history,
        num_workers=int(os.environ.get("BENCH_NUM_WORKERS") or 2),
        keep_period=keep_period or None,
        max_episodes=int(max_episodes) if max_episodes else None,
        gt_speed_input=gt_speed.enabled_from_env(),
        gt_speed_list=os.environ.get(gt_speed.LIST_ENV_VAR) or None,
        episode_uniform=episode_sampling.enabled_from_env(),
        space=action_space.from_env(),
        ee_dataset=os.environ.get(action_space.DATASET_ENV_VAR) or None,
    )
    install_history_preprocess(exterior_history if cfg.model.model_type == _model.ModelType.PI05 else None)
    _data_loader.create_torch_dataset = _create_torch_dataset
    _data_loader.create_torch_data_loader = _create_torch_data_loader
    _c._CONFIGS[:] = [c for c in _c._CONFIGS if c.name != cfg.name]
    _c._CONFIGS.append(cfg)
    _c._CONFIGS_DICT = {c.name: c for c in _c._CONFIGS}


def patch_checkpoint_vintage_stamp(checkpoints_module, schema: str | None) -> None:
    camera_vintage.patch_checkpoint_vintage_stamp(checkpoints_module, schema)


def assert_checkpoint_hook_reachable(train_module, checkpoints_module) -> None:
    camera_vintage.assert_checkpoint_hook_reachable(train_module, checkpoints_module)


def check_checkpoint_camera_vintage(ckpt_dir) -> str:
    expected = camera_vintage.expected_schema()
    camera_vintage.check_checkpoint_camera_vintage(ckpt_dir, expected=expected)
    return expected


def check_dataset_camera_vintage() -> str | None:
    repo_id = os.environ.get("BENCH_DATASET_REPO")
    if not repo_id:
        return None
    return camera_vintage.check_camera_vintage(repo_id)


def patch_checkpoint_save_limit(checkpoints_module, concurrent_gb: int) -> None:
    ocp = checkpoints_module.ocp
    handler_cls = ocp.PyTreeCheckpointHandler
    original = checkpoints_module.initialize_checkpoint_dir

    @functools.wraps(original)
    def patched(checkpoint_dir, **kwargs):
        built = []

        def limited(*args, **handler_kwargs):
            built.append(handler_cls(*args, save_concurrent_gb=concurrent_gb, **handler_kwargs))
            return built[-1]

        ocp.PyTreeCheckpointHandler = limited
        try:
            result = original(checkpoint_dir, **kwargs)
        finally:
            ocp.PyTreeCheckpointHandler = handler_cls
        if len(built) != 2:
            raise RuntimeError(
                f"expected upstream to build 2 PyTree handlers (train_state, params), built {len(built)}"
            )
        return result

    checkpoints_module.initialize_checkpoint_dir = patched


def patch_resume_to_current_devices(train_module, checkpoints_module) -> None:
    ocp = checkpoints_module.ocp
    original_init = train_module.init_train_state

    @functools.wraps(original_init)
    def init_train_state(config, init_rng, mesh, *, resume):
        state, state_sharding = original_init(config, init_rng, mesh, resume=resume)
        if resume:
            with at.disable_typechecking():
                state = jax.tree.map(
                    lambda x, sharding: jax.ShapeDtypeStruct(
                        x.shape, x.dtype, sharding=sharding, weak_type=x.weak_type
                    ),
                    state,
                    state_sharding,
                )
        return state, state_sharding

    def restore_args(item):
        return ocp.args.PyTreeRestore(item=item, restore_args=ocp.checkpoint_utils.construct_restore_args(item))

    @functools.wraps(checkpoints_module.restore_state)
    def restore_state(checkpoint_manager, state, data_loader, step=None):
        del data_loader
        with at.disable_typechecking():
            train_state, params = checkpoints_module._split_params(state)
            restored = checkpoint_manager.restore(
                step,
                args=ocp.args.Composite(train_state=restore_args(train_state), params=restore_args({"params": params})),
            )
        return checkpoints_module._merge_params(restored["train_state"], restored["params"])

    train_module.init_train_state = init_train_state
    checkpoints_module.restore_state = restore_state


def patch_checkpoint_history_stamp(checkpoints_module) -> None:
    offsets = history.offsets_from_env()
    original = checkpoints_module.initialize_checkpoint_dir

    @functools.wraps(original)
    def patched(checkpoint_dir, **kwargs):
        manager, resuming = original(checkpoint_dir, **kwargs)
        history.stamp_run_dir(pathlib.Path(str(checkpoint_dir)), offsets, existing_steps=tuple(manager.all_steps()))
        return manager, resuming

    checkpoints_module.initialize_checkpoint_dir = patched


def schedule_record(config: _c.TrainConfig) -> dict:
    schedule = config.lr_schedule
    if not dataclasses.is_dataclass(schedule) or isinstance(schedule, type):
        raise TypeError(f"cannot record the learning-rate schedule {schedule!r}")
    return {
        "num_train_steps": config.num_train_steps,
        "lr_schedule": {type(schedule).__name__: dataclasses.asdict(schedule)},
    }


def stamp_lr_schedule(run_dir: pathlib.Path, record: dict, *, existing_steps: tuple[int, ...]) -> None:
    path = run_dir / SCHEDULE_STAMP
    if path.is_file():
        recorded = json.loads(path.read_text())
        if recorded != json.loads(json.dumps(record)):
            raise RuntimeError(f"{path}: the run trains on {recorded}; this launch asks for {record}")
        return
    if existing_steps:
        raise RuntimeError(f"{run_dir} holds checkpoints {existing_steps} but no {SCHEDULE_STAMP}")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(path)


def patch_checkpoint_schedule_stamp(checkpoints_module, config: _c.TrainConfig) -> None:
    record = schedule_record(config)
    original = checkpoints_module.initialize_checkpoint_dir

    @functools.wraps(original)
    def patched(checkpoint_dir, **kwargs):
        manager, resuming = original(checkpoint_dir, **kwargs)
        stamp_lr_schedule(pathlib.Path(str(checkpoint_dir)), record, existing_steps=tuple(manager.all_steps()))
        return manager, resuming

    checkpoints_module.initialize_checkpoint_dir = patched


def check_checkpoint_history(ckpt_dir) -> None:
    history.check_serve(ckpt_dir, history.offsets_from_env())


def patch_checkpoint_gt_speed_stamp(checkpoints_module) -> None:
    enabled = gt_speed.enabled_from_env()
    list_path = os.environ.get(gt_speed.LIST_ENV_VAR) or None
    original = checkpoints_module.initialize_checkpoint_dir

    @functools.wraps(original)
    def patched(checkpoint_dir, **kwargs):
        manager, resuming = original(checkpoint_dir, **kwargs)
        gt_speed.stamp_run_dir(
            pathlib.Path(str(checkpoint_dir)), enabled, list_path, existing_steps=tuple(manager.all_steps())
        )
        return manager, resuming

    checkpoints_module.initialize_checkpoint_dir = patched


def patch_checkpoint_sampler_stamp(checkpoints_module) -> None:
    enabled = episode_sampling.enabled_from_env()
    original = checkpoints_module.initialize_checkpoint_dir

    @functools.wraps(original)
    def patched(checkpoint_dir, **kwargs):
        manager, resuming = original(checkpoint_dir, **kwargs)
        episode_sampling.stamp_run_dir(
            pathlib.Path(str(checkpoint_dir)), enabled, existing_steps=tuple(manager.all_steps())
        )
        return manager, resuming

    checkpoints_module.initialize_checkpoint_dir = patched


def check_checkpoint_gt_speed(ckpt_dir) -> None:
    gt_speed.check_serve(ckpt_dir, gt_speed.enabled_from_env())


def patch_checkpoint_action_space_stamp(checkpoints_module) -> None:
    space = action_space.from_env()
    ee_dataset = os.environ.get(action_space.DATASET_ENV_VAR) or None
    euler = (
        action_space.read_euler_convention(HF_LEROBOT_HOME / ee_dataset)
        if space != action_space.JOINT_POSITION and ee_dataset
        else None
    )
    original = checkpoints_module.initialize_checkpoint_dir

    @functools.wraps(original)
    def patched(checkpoint_dir, **kwargs):
        manager, resuming = original(checkpoint_dir, **kwargs)
        action_space.stamp_run_dir(
            pathlib.Path(str(checkpoint_dir)), space, ee_dataset, euler, existing_steps=tuple(manager.all_steps())
        )
        return manager, resuming

    checkpoints_module.initialize_checkpoint_dir = patched


def check_checkpoint_action_space(ckpt_dir) -> dict[str, str]:
    return action_space.check_serve(ckpt_dir, action_space.from_env())
