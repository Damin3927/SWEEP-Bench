SHELL := /bin/bash

VENV ?= $(CURDIR)/.venv
PY   := $(VENV)/bin/python

BASE_REPO     ?= openvla/openvla-7b
BASE_REVISION ?= 47a0ec7fc4ec123775a391911046cf33cf9ed83f
BASE          ?=

DATASET_ROOT ?=
STATS_DIR    ?=
RUN_DIR      ?=
HISTORY      ?=
MAX_EPISODES ?=

BATCH         ?= 4
ACCUM         ?= 16
MAX_STEPS     ?= 25000
DECAY_STEP    ?= 10000
ANNEAL_FROM   ?= 15000
ANNEAL_END    ?= 0.01
SAVE_INTERVAL ?= 1000
KEEP_INTERVAL ?= 5000
NUM_WORKERS   ?= 6
WANDB_PROJECT ?=
TRAIN_ARGS    ?=

STEP_DIR ?=
PORT     ?= 8000

.PHONY: check-submodule check-venv sync lock patch base prep train run clean

check-submodule:
	@if [ ! -f upstream/prismatic/vla/constants.py ]; then \
		echo "ERROR: upstream/ is empty. Run: git submodule update --init policy_servers/openvlaoft/upstream"; \
		echo "    or, without git metadata (e.g. from an archive): scripts/fetch_upstreams.sh"; \
		exit 1; \
	fi

check-venv:
	@if [ ! -x $(PY) ]; then echo "ERROR: $(VENV) is missing. Run 'make sync' first."; exit 1; fi

sync:
	UV_PROJECT_ENVIRONMENT=$(VENV) uv sync --project . --frozen

lock:
	uv lock --project .

patch: check-submodule
	@for p in $(CURDIR)/finetune/patches/*.patch; do \
		name=$$(basename $$p); \
		if git -C upstream apply --reverse --check "$$p" 2>/dev/null; then \
			echo "patch: $$name already applied, skipping"; \
		elif git -C upstream apply --check "$$p" 2>/dev/null; then \
			git -C upstream apply "$$p" || exit 1; \
			echo "patch: applied $$name"; \
		else \
			echo "ERROR: $$name applies neither forward nor in reverse to upstream/."; \
			exit 1; \
		fi; \
	done

base: check-venv
	@if [ -z "$(BASE)" ]; then echo "ERROR: set BASE=<dir> for the openvla-7b snapshot"; exit 1; fi
	$(PY) -c "from huggingface_hub import snapshot_download as s; print(s('$(BASE_REPO)', revision='$(BASE_REVISION)', local_dir='$(BASE)'))"

prep: check-venv patch
	@if [ -z "$(DATASET_ROOT)" ] || [ -z "$(STATS_DIR)" ]; then echo "ERROR: set DATASET_ROOT and STATS_DIR"; exit 1; fi
	$(PY) finetune/prep.py --dataset-root $(DATASET_ROOT) --out $(STATS_DIR) \
		$(if $(MAX_EPISODES),--max-episodes $(MAX_EPISODES))

train: check-venv patch
	@if [ -z "$(DATASET_ROOT)" ] || [ -z "$(STATS_DIR)" ] || [ -z "$(BASE)" ] || [ -z "$(RUN_DIR)" ]; then \
		echo "ERROR: set DATASET_ROOT, STATS_DIR, BASE and RUN_DIR"; exit 1; \
	fi
	BENCH_EXTERIOR_HISTORY=$(HISTORY) $(PY) finetune/train.py \
		--dataset-root $(DATASET_ROOT) --stats $(STATS_DIR)/dataset_statistics.json --base $(BASE) \
		--run-dir $(RUN_DIR) --batch-size $(BATCH) --grad-accum $(ACCUM) --max-steps $(MAX_STEPS) \
		--decay-step $(DECAY_STEP) $(if $(ANNEAL_FROM),--anneal-from $(ANNEAL_FROM) --anneal-end $(ANNEAL_END)) \
		--save-interval $(SAVE_INTERVAL) --keep-interval $(KEEP_INTERVAL) \
		--num-workers $(NUM_WORKERS) $(if $(MAX_EPISODES),--max-episodes $(MAX_EPISODES)) \
		$(if $(WANDB_PROJECT),--wandb-project $(WANDB_PROJECT)) $(TRAIN_ARGS)

run: check-venv patch
	@if [ -z "$(RUN_DIR)" ] || [ -z "$(BASE)" ]; then echo "ERROR: set RUN_DIR and BASE"; exit 1; fi
	BENCH_EXTERIOR_HISTORY=$(HISTORY) $(PY) serve.py --run-dir $(RUN_DIR) --base $(BASE) --port $(PORT) \
		$(if $(STEP_DIR),--step-dir $(STEP_DIR))

clean:
	rm -rf $(VENV)
