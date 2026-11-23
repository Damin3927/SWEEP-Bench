SHELL := /bin/bash

VENV ?= .venv
PY := $(VENV)/bin/python
CUDA_SHIM := $(abspath $(VENV))/lib/bench-cuda
export UV_PROJECT_ENVIRONMENT := $(abspath $(VENV))
export LD_LIBRARY_PATH := $(CUDA_SHIM)$(if $(LD_LIBRARY_PATH),:$(LD_LIBRARY_PATH))

DATASET_DIR ?=
ASSETS_DIR ?=
HISTORY ?=
RUN ?=
OUTPUT_ROOT ?= $(abspath runs)
WORKERS ?= 8
CHECKPOINT ?=
PORT ?= 8000
WANDB ?= offline

.PHONY: sync lock assets train run clean check-venv

check-venv:
	@if [ ! -x $(PY) ]; then echo "ERROR: $(VENV) is missing. Run 'make sync' first."; exit 1; fi

sync:
	uv sync --project . --frozen
	mkdir -p $(CUDA_SHIM)
	ln -sf $$(ls $(abspath $(VENV))/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 \
		$(abspath $(VENV))/lib/python3.10/site-packages/torch/lib/libcudart.so.12 2>/dev/null | head -1) $(CUDA_SHIM)/libcudart.so
	test -e $(CUDA_SHIM)/libcudart.so

lock:
	uv lock --project .

assets: check-venv
	$(PY) -m finetune.compute_assets --dataset-dir $(DATASET_DIR) --assets-dir $(ASSETS_DIR) --stats --t5

train: check-venv
	@if [ -z "$(RUN)" ]; then echo "ERROR: set RUN (the run name under $(OUTPUT_ROOT)/cosmos_policy/sweep_bench/)"; exit 1; fi
	BENCH_DATASET_DIR=$(DATASET_DIR) BENCH_ASSETS_DIR=$(ASSETS_DIR) BENCH_EXTERIOR_HISTORY=$(HISTORY) \
	IMAGINAIRE_OUTPUT_ROOT=$(OUTPUT_ROOT) \
	$(VENV)/bin/torchrun --nproc_per_node=1 --master_port=$${MASTER_PORT:-12341} -m finetune.train \
		--config=finetune/bench_config.py -- experiment=cosmos_predict2_2b_480p_sweep_bench job.name=$(RUN) \
		dataloader_train.num_workers=$(WORKERS) job.wandb_mode=$(WANDB) $(EXTRA)

run: check-venv
	@if [ -z "$(CHECKPOINT)" ]; then echo "ERROR: set CHECKPOINT"; exit 1; fi
	BENCH_EXTERIOR_HISTORY=$(HISTORY) $(PY) serve.py --checkpoint-dir $(CHECKPOINT) --assets-dir $(ASSETS_DIR) --port $(PORT) \
		$(if $(T5_LIVE),--t5-live)

clean:
	rm -rf $(VENV)
