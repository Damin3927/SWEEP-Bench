SHELL := /bin/bash

DVLA_DIR ?= upstream

FFMPEG_URL ?= https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-shared-7.1.tar.xz

PRETRAINED   ?= ../checkpoints/dynamic-vla-DOM
PRETRAINED_REVISION ?= c9609d94c5940eb42436a31927d7e80ae454279e
DATASET_REPO ?=
EXP_NAME     ?= $(notdir $(DATASET_REPO))
NPROC        ?= 1

CONFIG       ?= ../finetune/hist_unfrozen.yaml
TRAIN_STEM   ?= 0
VIDEO_BACKEND ?= torchcodec
FFMPEG_DEP   := $(if $(filter pyav,$(VIDEO_BACKEND)),,ffmpeg)

TEST_FRACTION ?= 0.01
SPLIT_SEED    ?= 0

STATE_EVERY ?= 0
RESUME      ?= 0

SRC_REPO      ?=
DST_REPO      ?=
CONVERT_FLAGS ?=

.PHONY: sync patch ffmpeg pretrained convert-data train run clean check-submodule check-venv

check-submodule:
	@if [ ! -f $(DVLA_DIR)/run.py ]; then \
		echo "ERROR: $(DVLA_DIR)/ is empty. Initialize the submodule first:"; \
		echo "    git submodule update --init --recursive"; \
		echo "    or, without git metadata (e.g. from an archive): scripts/fetch_upstreams.sh"; \
		exit 1; \
	fi

check-venv: check-submodule
	@if [ ! -x $(DVLA_DIR)/.venv/bin/python ]; then \
		echo "ERROR: $(DVLA_DIR)/.venv is missing. Run 'make sync' first."; \
		exit 1; \
	fi

patch: check-submodule
	@for p in $(CURDIR)/finetune/patches/*.patch; do \
		name=$$(basename $$p); \
		if git -C $(DVLA_DIR) apply --reverse --check "$$p" 2>/dev/null; then \
			echo "patch: $$name already applied, skipping"; \
		elif git -C $(DVLA_DIR) apply --check "$$p" 2>/dev/null; then \
			git -C $(DVLA_DIR) apply "$$p" || exit 1; \
			echo "patch: applied $$name"; \
		else \
			echo "ERROR: $$name applies neither forward nor in reverse to $(DVLA_DIR)/."; \
			echo "       Regenerate finetune/patches/$$name against the current pin."; \
			exit 1; \
		fi; \
	done

sync: check-submodule patch
	cd $(DVLA_DIR) && (test -x .venv/bin/python || uv venv --no-config --python 3.10 .venv)
	cd $(DVLA_DIR) && uv pip install --no-config --python .venv/bin/python \
		torch==2.7.1 torchvision==0.22.1
	cd $(DVLA_DIR) && uv pip install --no-config --python .venv/bin/python \
		-r <(grep -v '^pylibmc' requirements.txt)
	cd $(DVLA_DIR) && uv pip install --no-config --python .venv/bin/python \
		websockets msgpack
	cd $(DVLA_DIR) && ln -sf \
		python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 \
		.venv/lib/libcudart.so
	$(if $(FFMPEG_DEP),$(MAKE) ffmpeg)

ffmpeg: .ffmpeg/lib/libavutil.so.59

.ffmpeg/lib/libavutil.so.59:
	mkdir -p .ffmpeg
	curl -fL $(FFMPEG_URL) | tar -xJ -C .ffmpeg --strip-components=1

pretrained: check-venv
	cd $(DVLA_DIR) && ./.venv/bin/hf download hzxie/dynamic-vla-DOM --revision $(PRETRAINED_REVISION) \
		--local-dir ../checkpoints/dynamic-vla-DOM

convert-data:
	@if [ -z "$(SRC_REPO)" ] || [ -z "$(DST_REPO)" ]; then \
		echo "ERROR: SRC_REPO and DST_REPO are required, e.g.:"; \
		echo "    make convert-data SRC_REPO=<hf-user>/<droid-dataset> DST_REPO=<hf-user>/<name>_ee_dom"; \
		exit 1; \
	fi
	cd ../.. && uv run --no-sync python \
		policy_servers/dynamicvla/finetune/convert_droid_dataset.py \
		--src $(SRC_REPO) --dst $(DST_REPO) $(CONVERT_FLAGS)

train: check-venv $(FFMPEG_DEP) patch
	@if [ -z "$(DATASET_REPO)" ]; then \
		echo "ERROR: DATASET_REPO is required, e.g.:"; \
		echo "    make train DATASET_REPO=<hf-user>/<dataset>_ee_dom"; \
		exit 1; \
	fi
	cd $(DVLA_DIR) && \
		PYTHONPATH=../shims:$$PYTHONPATH \
		LD_LIBRARY_PATH=$$PWD/../.ffmpeg/lib:$$PWD/.venv/lib:$$LD_LIBRARY_PATH \
		BENCH_TEST_FRACTION=$(TEST_FRACTION) BENCH_SPLIT_SEED=$(SPLIT_SEED) \
		BENCH_TRAIN_STEM=$(TRAIN_STEM) BENCH_VIDEO_BACKEND=$(VIDEO_BACKEND) \
		BENCH_STATE_EVERY=$(STATE_EVERY) BENCH_RESUME=$(RESUME) \
		./.venv/bin/torchrun --standalone --nproc_per_node=$(NPROC) ../finetune/train.py \
			-c $(CONFIG) -d $(DATASET_REPO) -p $(PRETRAINED) \
			-e $(EXP_NAME)

CKPT ?=
PORT ?= 8000
CUDA_GRAPH   ?= 0
WARM_PROMPTS ?=
FRAME_PUSH   ?= 0

run: check-venv
	@if [ -z "$(CKPT)" ]; then \
		echo "ERROR: CKPT is required, e.g.:"; \
		echo "    make run CKPT=upstream/runs/checkpoints/<EXP_NAME>"; \
		exit 1; \
	fi
	LD_LIBRARY_PATH=$$PWD/$(DVLA_DIR)/.venv/lib:$$LD_LIBRARY_PATH \
		./$(DVLA_DIR)/.venv/bin/python serve.py --checkpoint-dir $(CKPT) --port $(PORT) \
		$(if $(filter 1,$(CUDA_GRAPH)),--cuda-graph $(if $(WARM_PROMPTS),--warm-prompts $(WARM_PROMPTS))) \
		$(if $(filter 1,$(FRAME_PUSH)),--frame-push)

clean:
	rm -rf $(DVLA_DIR)/.venv
