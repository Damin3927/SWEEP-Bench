SHELL := /bin/bash

VENV       ?= .venv
PY         := $(VENV)/bin/python
BASE_MODEL ?= nvidia/GR00T-N1.7-3B
BASE_REVISION ?= 2fc962b973bccdd5d8ce4f67cc63b264d6886495
SOURCE     ?=
DATASET    ?=
RUN_DIR    ?=
CHECKPOINT ?=
PORT       ?= 8000
HISTORY    ?=
MAX_EPISODES ?=

MAX_STEPS  ?= 90000
RESTART_STEP   ?= 60000
RESTART_WARMUP ?= 1000
RESTART_PEAK   ?= 0.3
BATCH      ?= 128
GRAD_ACCUM ?= 1
SAVE_STEPS ?= 1000
NUM_WORKERS ?= 4
TRAIN_EXTRA ?=

LFS_WHEEL := upstream/scripts/deployment/dgpu/wheels/torchcodec-0.8.0-cp312-cp312-linux_aarch64.whl
LFS_URL   := https://media.githubusercontent.com/media/NVIDIA/Isaac-GR00T/$(shell git -C upstream rev-parse HEAD 2>/dev/null)/$(patsubst upstream/%,%,$(LFS_WHEEL))

.PHONY: check-submodule fetch-lfs sync prep train run

check-submodule:
	@if [ ! -f upstream/pyproject.toml ]; then \
		echo "ERROR: upstream/ is empty. Run: git submodule update --init policy_servers/groot/upstream"; \
		echo "    or, without git metadata (e.g. from an archive): scripts/fetch_upstreams.sh"; \
		exit 1; \
	fi

fetch-lfs: check-submodule
	@oid=$$(sed -n 's/^oid sha256://p' $(LFS_WHEEL)); \
	if [ -n "$$oid" ]; then \
		curl -fsSL -o $(LFS_WHEEL).tmp $(LFS_URL) && \
		echo "$$oid  $(LFS_WHEEL).tmp" | sha256sum -c - && mv $(LFS_WHEEL).tmp $(LFS_WHEEL); \
	fi

sync: fetch-lfs
	UV_PROJECT_ENVIRONMENT=$(VENV) uv sync --frozen

prep:
	@if [ -z "$(SOURCE)" ] || [ -z "$(DATASET)" ]; then echo "ERROR: set SOURCE and DATASET"; exit 1; fi
	$(PY) finetune/prep_dataset.py --source $(SOURCE) --out $(DATASET) \
		$(if $(MAX_EPISODES),--max-episodes $(MAX_EPISODES))

train:
	@if [ -z "$(DATASET)" ] || [ -z "$(RUN_DIR)" ]; then echo "ERROR: set DATASET and RUN_DIR"; exit 1; fi
	base=$$($(PY) -c "from huggingface_hub import snapshot_download as s; \
		print(s('$(BASE_MODEL)', revision='$(BASE_REVISION)'))") && \
	BENCH_EXTERIOR_HISTORY=$(HISTORY) BENCH_RESTART_STEP=$(RESTART_STEP) BENCH_RESTART_WARMUP=$(RESTART_WARMUP) \
	BENCH_RESTART_PEAK=$(RESTART_PEAK) $(PY) finetune/train.py \
		--base-model-path $$base --dataset-path $(DATASET) --output-dir $(RUN_DIR) \
		--max-steps $(MAX_STEPS) --global-batch-size $(BATCH) --gradient-accumulation-steps $(GRAD_ACCUM) \
		--save-steps $(SAVE_STEPS) --save-total-limit 3 --learning-rate 1e-4 --warmup-ratio 0.05 \
		--weight-decay 1e-5 --state-dropout-prob 0.2 --dataloader-num-workers $(NUM_WORKERS) \
		--color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 $(TRAIN_EXTRA)

run:
	@if [ -z "$(CHECKPOINT)" ]; then echo "ERROR: set CHECKPOINT=<run dir>/checkpoint-<step>"; exit 1; fi
	BENCH_EXTERIOR_HISTORY=$(HISTORY) $(PY) serve.py --checkpoint $(CHECKPOINT) --port $(PORT)
