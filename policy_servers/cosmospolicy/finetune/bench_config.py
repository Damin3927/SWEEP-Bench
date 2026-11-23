from __future__ import annotations

import copy
import math
import os
import sys
import types
from typing import TYPE_CHECKING

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.utils import checkpoint_db
from cosmos_policy.config.config import make_config as make_upstream_config
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE
from hydra.core.config_store import ConfigStore

from .bench_dataset import BenchLeRobotDataset
from .history import LatentLayout, offsets_from_env

if TYPE_CHECKING:
    from types import ModuleType

EXPERIMENT_NAME = "cosmos_predict2_2b_480p_sweep_bench"
INFERENCE_EXPERIMENT_NAME = f"{EXPERIMENT_NAME}__inference"
CHUNK_SIZE = 16
GLOBAL_BATCH = 128
LOCAL_BATCH = 16
SAVE_EVERY = 250
TRAIN_FRAMES = 1_259_153
EPOCH_STEPS = math.ceil(TRAIN_FRAMES / GLOBAL_BATCH)
WARMUP_STEPS = math.ceil(EPOCH_STEPS / 30 / 50) * 50
ANNEAL_STEPS = 6000
JOB_GROUP = "sweep_bench"
BASE_REPO = "nvidia/Cosmos-Predict2-2B-Video2World"
BASE_FILE = "model-480p-16fps.pt"
BASE_REVISION = "f50c09f5d8ab133a90cac3f4886a6471e9ba3f18"
DEFAULT_BASE_CKPT = f"hf://{BASE_REPO}/{BASE_FILE}"
DEFAULT_VAE = "hf://Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"


def _optional_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


def _anneal_after_first_cycle(scheduler: LazyDict, steps: int) -> None:
    end = scheduler["f_min"][1]
    first_end = scheduler["f_min"][0]
    scheduler["cycle_lengths"] = [scheduler["cycle_lengths"][0], steps, scheduler["cycle_lengths"][1]]
    scheduler["warm_up_steps"] = [scheduler["warm_up_steps"][0], 0, 0]
    scheduler["f_start"] = [scheduler["f_start"][0], first_end, end]
    scheduler["f_max"] = [scheduler["f_max"][0], first_end, end]
    scheduler["f_min"] = [first_end, end, end]


def build_experiments(upstream_experiments: ModuleType) -> tuple[LazyDict, LazyDict]:
    history = offsets_from_env()
    layout = LatentLayout(history)
    dataset = L(BenchLeRobotDataset)(
        data_dir=os.environ.get("BENCH_DATASET_DIR", ""),
        assets_dir=os.environ.get("BENCH_ASSETS_DIR", ""),
        history=history,
        chunk_size=CHUNK_SIZE,
        use_image_aug=True,
        use_stronger_image_aug=True,
        gamma=0.99,
        max_episodes=_optional_int("BENCH_MAX_EPISODES"),
    )
    train = copy.deepcopy(upstream_experiments.cosmos_predict2_2b_480p_libero)
    model_config = train["model"]["config"]
    model_config["state_t"] = layout.state_t
    model_config["min_num_conditional_frames"] = layout.num_conditional_frames
    model_config["max_num_conditional_frames"] = layout.num_conditional_frames
    model_config["tokenizer"]["chunk_duration"] = layout.num_video_frames
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_batch = _optional_int("BENCH_LOCAL_BATCH") or LOCAL_BATCH
    if GLOBAL_BATCH % (local_batch * world):
        raise ValueError(
            f"WORLD_SIZE={world} x BENCH_LOCAL_BATCH={local_batch} does not divide the global batch {GLOBAL_BATCH}"
        )
    model_config["fsdp_shard_size"] = world
    model_config["tokenizer"]["vae_pth"] = os.environ.get("BENCH_VAE_PTH", DEFAULT_VAE)
    train["checkpoint"]["load_path"] = os.environ.get("BENCH_BASE_CKPT", DEFAULT_BASE_CKPT)
    train["dataloader_train"]["dataset"] = dataset
    train["dataloader_train"]["sampler"]["dataset"] = dataset
    train["dataloader_train"]["batch_size"] = local_batch
    train["trainer"]["grad_accum_iter"] = GLOBAL_BATCH // (local_batch * world)
    train["trainer"]["max_iter"] = EPOCH_STEPS + ANNEAL_STEPS
    train["checkpoint"]["save_iter"] = SAVE_EVERY
    train["scheduler"]["cycle_lengths"] = [EPOCH_STEPS, 100000000000000]
    train["scheduler"]["warm_up_steps"] = [WARMUP_STEPS, 0]
    _anneal_after_first_cycle(train["scheduler"], ANNEAL_STEPS)
    train["job"]["group"] = JOB_GROUP
    train["job"]["name"] = EXPERIMENT_NAME
    inference = LazyDict(
        dict(
            defaults=[f"/experiment/{EXPERIMENT_NAME}", "_self_"],
            model=L(CosmosPolicyVideo2WorldModel)(config=dict(sde=L(HybridEDMSDE)(sigma_max=80, sigma_min=4))),
            job=dict(group=f"{JOB_GROUP}_inference", name=INFERENCE_EXPERIMENT_NAME),
        )
    )
    return train, inference


def _allow_unloadable_decord() -> None:
    try:
        import decord  # noqa: F401
    except OSError:
        sys.modules["decord"] = types.ModuleType("decord")


def make_config():
    _allow_unloadable_decord()
    resolve = checkpoint_db.get_checkpoint_path
    checkpoint_db.get_checkpoint_path = str
    try:
        config = make_upstream_config()
    finally:
        checkpoint_db.get_checkpoint_path = resolve
    store = ConfigStore.instance()
    from cosmos_policy.config.experiment import cosmos_policy_experiment_configs

    for item in build_experiments(cosmos_policy_experiment_configs):
        store.store(group="experiment", package="_global_", name=item["job"]["name"], node=item)
    return config
