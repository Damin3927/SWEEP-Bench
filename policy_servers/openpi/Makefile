OPENPI_DIR        ?= upstream
OPENPI_CONFIG     ?= bench
OPENPI_CHECKPOINT ?=
HOST_PORT         ?= 8000
FAST              ?=
FAST_HOST         ?=
JAX_CACHE_DIR     ?=
BATCHED_SIGLIP    ?=
FRAME_PUSH        ?=
BELT_SPEED_ADAPTER ?=

DATASET_REPO ?=
POLICY       ?= pi05_base
ifeq ($(POLICY),pi05_base)
NUM_STEPS    ?= 30000
EXP_NAME     ?= $(notdir $(DATASET_REPO))
else
NUM_STEPS    ?= 130000
EXP_NAME     ?= $(POLICY)_$(notdir $(DATASET_REPO))
endif
MAX_FRAMES   ?= 1000000
FSDP_DEVICES ?= 1
HISTORY      ?=
GT_SPEED      ?=
GT_SPEED_LIST ?=
EPISODE_UNIFORM ?=
ACTION_SPACE ?=
EE_DATASET   ?=
NUM_WORKERS  ?= 2
KEEP_PERIOD  ?= 1000
MAX_EPISODES ?=

_BENCH_ENV = BENCH_DATASET_REPO=$(DATASET_REPO) BENCH_NUM_STEPS=$(NUM_STEPS) BENCH_POLICY=$(POLICY) BENCH_FSDP_DEVICES=$(FSDP_DEVICES) BENCH_EXTERIOR_HISTORY=$(HISTORY) BENCH_NUM_WORKERS=$(NUM_WORKERS) BENCH_KEEP_PERIOD=$(KEEP_PERIOD) BENCH_MAX_EPISODES=$(MAX_EPISODES) BENCH_GT_SPEED=$(GT_SPEED) BENCH_GT_SPEED_LIST=$(GT_SPEED_LIST) BENCH_EPISODE_UNIFORM=$(EPISODE_UNIFORM) BENCH_ACTION_SPACE=$(ACTION_SPACE) BENCH_EE_DATASET=$(EE_DATASET)

.PHONY: sync run clean check-submodule train sync-convert convert-torch

check-submodule:
	@if [ ! -f $(OPENPI_DIR)/pyproject.toml ]; then \
		echo "ERROR: $(OPENPI_DIR)/ is empty. Initialize the submodule first:"; \
		echo "    git submodule update --init --recursive"; \
		echo "    or, without git metadata (e.g. from an archive): scripts/fetch_upstreams.sh"; \
		exit 1; \
	fi

sync: check-submodule
	cd $(OPENPI_DIR) && GIT_LFS_SKIP_SMUDGE=1 uv sync
	cd $(OPENPI_DIR) && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

run: check-submodule
	@if [ -z "$(OPENPI_CHECKPOINT)" ]; then \
		echo "ERROR: OPENPI_CHECKPOINT is empty. Pass a trained checkpoint dir, e.g.:"; \
		echo "    make run DATASET_REPO=<hf-user>/<dataset> OPENPI_CHECKPOINT=<path>/upstream/checkpoints/bench/<exp>/<step>"; \
		exit 1; \
	fi
	@if [ "$(OPENPI_CONFIG)" = "bench" ] && [ -z "$(DATASET_REPO)" ]; then \
		echo "ERROR: serving the bench config needs DATASET_REPO (to rebuild the model"; \
		echo "       variant and locate the checkpoint's norm-stats). Example:"; \
		echo "    make run DATASET_REPO=<hf-user>/<dataset> OPENPI_CHECKPOINT=<...>"; \
		echo "  Or serve a stock openpi config: make run OPENPI_CONFIG=pi05_libero OPENPI_CHECKPOINT=gs://..."; \
		exit 1; \
	fi
	cd $(OPENPI_DIR) && \
		PYTHONFAULTHANDLER=1 \
		XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
		$(if $(JAX_PLATFORMS),JAX_PLATFORMS=$(JAX_PLATFORMS)) \
		$(if $(FAST_HOST),BENCH_FAST_HOST=$(FAST_HOST)) \
		$(if $(BATCHED_SIGLIP),BENCH_BATCHED_SIGLIP=$(BATCHED_SIGLIP)) \
		$(if $(JAX_CACHE_DIR),JAX_COMPILATION_CACHE_DIR=$(abspath $(JAX_CACHE_DIR))) \
		$(_BENCH_ENV) \
		./.venv/bin/python ../serve.py \
			--config=$(OPENPI_CONFIG) \
			--checkpoint-dir=$(OPENPI_CHECKPOINT) \
			--port=$(HOST_PORT) \
			$(if $(FAST),--fast=$(FAST)) \
			$(if $(FRAME_PUSH),--frame-push) \
			$(if $(BELT_SPEED_ADAPTER),--belt-speed-adapter=$(abspath $(BELT_SPEED_ADAPTER)) --belt-speed-adapter-cuda-graph)

clean:
	rm -rf $(OPENPI_DIR)/.venv


TRAIN_ARGS ?=
TRAIN_EXTRA :=
ifeq ($(RESUME),1)
TRAIN_EXTRA += --resume
endif
ifeq ($(OVERWRITE),1)
TRAIN_EXTRA += --overwrite
endif

train: check-submodule
	@if [ -z "$(DATASET_REPO)" ]; then \
		echo "ERROR: DATASET_REPO is required, e.g.:"; \
		echo "    make train DATASET_REPO=<hf-user>/<dataset> NUM_STEPS=30000"; \
		exit 1; \
	fi
	cd $(OPENPI_DIR) && \
		$(_BENCH_ENV) \
		./.venv/bin/python ../finetune/compute_norm_stats.py \
			--config-name=bench --max-frames=$(MAX_FRAMES)
	cd $(OPENPI_DIR) && \
		PYTHONFAULTHANDLER=1 \
		XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
		$(_BENCH_ENV) \
		./.venv/bin/python ../finetune/train.py \
			bench --exp-name=$(EXP_NAME) $(TRAIN_EXTRA) $(TRAIN_ARGS)

CONVERT_VENV     ?= $(CURDIR)/.venv-convert
TORCH_CHECKPOINT ?=

sync-convert: check-submodule
	cd $(OPENPI_DIR) && UV_PROJECT_ENVIRONMENT=$(CONVERT_VENV) GIT_LFS_SKIP_SMUDGE=1 uv sync
	cp -r $(OPENPI_DIR)/src/openpi/models_pytorch/transformers_replace/* \
		$(CONVERT_VENV)/lib/python3.11/site-packages/transformers/

convert-torch: check-submodule
	@if [ -z "$(OPENPI_CHECKPOINT)" ] || [ -z "$(TORCH_CHECKPOINT)" ] || [ -z "$(DATASET_REPO)" ]; then \
		echo "ERROR: convert-torch needs OPENPI_CHECKPOINT (JAX step dir), TORCH_CHECKPOINT (new run dir)"; \
		echo "       and the training run's DATASET_REPO / HISTORY."; \
		exit 1; \
	fi
	cd $(OPENPI_DIR) && \
		JAX_PLATFORMS=cpu \
		$(_BENCH_ENV) \
		$(CONVERT_VENV)/bin/python ../finetune/convert_torch.py \
			--checkpoint-dir=$(OPENPI_CHECKPOINT) \
			--output-dir=$(TORCH_CHECKPOINT)
