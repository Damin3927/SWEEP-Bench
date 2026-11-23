SHELL := /bin/bash

VENV        ?= .venv
WEIGHTS_DIR ?= checkpoints
PRETRAINED  ?= $(WEIGHTS_DIR)/VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt
QWEN        ?= $(WEIGHTS_DIR)/Qwen3-VL-2B-Instruct
VJEPA       ?= $(WEIGHTS_DIR)/vjepa2-vitl-fpc64-256
QWEN_REVISION    ?= 89644892e4d85e24eaac8bacfd4f463576704203
VJEPA_REVISION   ?= b3c1679b7c34d3255ef3547f27c7b226aefab26f
VLAJEPA_REVISION ?= db32b4a4457c39921869f667ac76cf8a8fc39da8

DATA_SRC    ?=
DATA_ROOT   ?=
RUN_DIR     ?=
HISTORY     ?=
STEPS       ?= 45000
MICRO_BATCH ?= 16
GRAD_ACCUM  ?= 4
WORKERS     ?= 8
TRAIN_FLAGS ?=
SERVE_FLAGS ?=
PORT        ?= 8000

PY  := UV_PROJECT_ENVIRONMENT=$(abspath $(VENV)) PYTHONPATH=$(CURDIR)/upstream TOKENIZERS_PARALLELISM=false \
	$(abspath $(VENV))/bin/python

.PHONY: sync lock patch weights prepare train run clean check-submodule check-venv

check-submodule:
	@if [ ! -f upstream/starVLA/model/framework/VLA_JEPA.py ]; then \
		echo "ERROR: upstream/ is empty. Run: git submodule update --init policy_servers/vlajepa/upstream"; \
		echo "    or, without git metadata (e.g. from an archive): scripts/fetch_upstreams.sh"; \
		exit 1; \
	fi

check-venv:
	@if [ ! -x $(VENV)/bin/python ]; then echo "ERROR: $(VENV) is missing. Run 'make sync' first."; exit 1; fi

sync:
	UV_PROJECT_ENVIRONMENT=$(abspath $(VENV)) uv sync --project . --frozen

lock:
	UV_PROJECT_ENVIRONMENT=$(abspath $(VENV)) uv lock --project .

patch: check-submodule
	@for p in $(CURDIR)/finetune/patches/*.patch; do \
		name=$$(basename $$p); \
		if git -C upstream apply --reverse --check "$$p" 2>/dev/null; then \
			echo "patch: $$name already applied, skipping"; \
		elif git -C upstream apply --check "$$p" 2>/dev/null; then \
			git -C upstream apply "$$p" && echo "patch: applied $$name"; \
		else \
			echo "ERROR: $$name applies neither forward nor in reverse to upstream/."; exit 1; \
		fi; \
	done

weights: check-venv
	$(PY) -c "from huggingface_hub import snapshot_download as s; \
		s('Qwen/Qwen3-VL-2B-Instruct', revision='$(QWEN_REVISION)', local_dir='$(QWEN)'); \
		s('facebook/vjepa2-vitl-fpc64-256', revision='$(VJEPA_REVISION)', local_dir='$(VJEPA)', \
			ignore_patterns=['original/*', '*.ipynb']); \
		s('ginwind/VLA-JEPA', revision='$(VLAJEPA_REVISION)', local_dir='$(WEIGHTS_DIR)/VLA-JEPA', \
			allow_patterns=['Pretrain/*', 'README.md'])"

prepare: check-venv
	@if [ -z "$(DATA_SRC)" ] || [ -z "$(DATA_ROOT)" ]; then echo "ERROR: set DATA_SRC and DATA_ROOT"; exit 1; fi
	cd finetune && $(PY) bench_data.py --src $(DATA_SRC) --out $(abspath $(DATA_ROOT)) --workers $(WORKERS)

train: check-venv patch
	@if [ -z "$(DATA_ROOT)" ] || [ -z "$(RUN_DIR)" ]; then echo "ERROR: set DATA_ROOT and RUN_DIR"; exit 1; fi
	BENCH_EXTERIOR_HISTORY=$(HISTORY) $(PY) finetune/train.py --data-root $(abspath $(DATA_ROOT)) \
		--run-dir $(abspath $(RUN_DIR)) --pretrained $(abspath $(PRETRAINED)) --qwen $(abspath $(QWEN)) \
		--vjepa $(abspath $(VJEPA)) --steps $(STEPS) --micro-batch $(MICRO_BATCH) --grad-accum $(GRAD_ACCUM) \
		--workers $(WORKERS) $(TRAIN_FLAGS)

run: check-venv patch
	@if [ -z "$(RUN_DIR)" ]; then echo "ERROR: set RUN_DIR"; exit 1; fi
	BENCH_EXTERIOR_HISTORY=$(HISTORY) $(PY) serve.py --run-dir $(abspath $(RUN_DIR)) \
		--qwen $(abspath $(QWEN)) --vjepa $(abspath $(VJEPA)) --port $(PORT) $(SERVE_FLAGS)

clean:
	rm -rf $(VENV)
