SHELL := /bin/bash

UPSTREAM := upstream
FASTWAM_DIR ?= $(abspath $(UPSTREAM))
VENV := .venv
PATCHES := $(sort $(wildcard $(CURDIR)/patches/*.patch))
PATCH_DEP := $(if $(filter $(abspath $(UPSTREAM)),$(abspath $(FASTWAM_DIR))),patch)

RUN_DIR ?=
WEIGHTS ?=
PORT    ?= 8000
CUDA_GRAPH ?=
FRAME_PUSH ?=
SERVE_ARGS = --run-dir $(RUN_DIR) $(if $(WEIGHTS),--weights $(WEIGHTS)) --fastwam-dir $(FASTWAM_DIR)

.PHONY: sync lock patch run clean check-venv check-fastwam check-run-dir check-submodule

check-submodule:
	@if [ ! -f $(UPSTREAM)/src/fastwam/models/wan22/fastwam.py ]; then \
		echo "ERROR: $(UPSTREAM)/ is empty. Initialize the submodule first:"; \
		echo "    git submodule update --init policy_servers/fastwam/upstream"; \
		echo "    or, without git metadata (e.g. from an archive): scripts/fetch_upstreams.sh"; \
		exit 1; \
	fi

patch: check-submodule
	@want=$$(cat $(PATCHES) | git -C $(UPSTREAM) patch-id --stable | cut -d' ' -f1); \
	have=$$(git -C $(UPSTREAM) log --reverse -p --format='commit %H' -n $(words $(PATCHES)) HEAD \
		| git -C $(UPSTREAM) patch-id --stable | cut -d' ' -f1); \
	if [ "$$want" = "$$have" ]; then echo "patch: already applied"; exit 0; fi; \
	if [ -n "$$(git -C $(UPSTREAM) status --porcelain)" ]; then \
		echo "ERROR: $(UPSTREAM)/ has local changes; reset it to the pinned commit first:"; \
		echo "    git submodule update --force policy_servers/fastwam/upstream"; \
		echo "    or, without git metadata: rm -rf policy_servers/fastwam/upstream && scripts/fetch_upstreams.sh"; \
		exit 1; \
	fi; \
	if ! git -C $(UPSTREAM) -c user.name=bench -c user.email=bench@localhost am --quiet --whitespace=nowarn $(PATCHES); then \
		git -C $(UPSTREAM) -c user.name=bench -c user.email=bench@localhost am --abort; \
		echo "ERROR: patches/ do not apply to $(UPSTREAM)/. If it holds an older version of the series, reset it"; \
		echo "       to the pinned commit and rerun make patch:"; \
		echo "    git submodule update --force policy_servers/fastwam/upstream"; \
		echo "    or, without git metadata: rm -rf policy_servers/fastwam/upstream && scripts/fetch_upstreams.sh"; \
		exit 1; \
	fi; \
	echo "patch: applied $(words $(PATCHES)) patches"

check-fastwam:
	@if [ ! -f $(FASTWAM_DIR)/src/fastwam/models/wan22/fastwam.py ]; then \
		echo "ERROR: FASTWAM_DIR=$(FASTWAM_DIR) does not look like a FastWAM checkout."; \
		exit 1; \
	fi

check-venv:
	@if [ ! -x $(VENV)/bin/python ]; then \
		echo "ERROR: $(VENV) is missing. Run 'make sync' first."; \
		exit 1; \
	fi

sync:
	uv sync --project . --frozen

lock:
	uv lock --project .

run: check-venv $(PATCH_DEP) check-fastwam check-run-dir
	./$(VENV)/bin/python serve.py $(SERVE_ARGS) --port $(PORT) $(if $(CUDA_GRAPH),--cuda-graph) $(if $(FRAME_PUSH),--frame-push)

check-run-dir:
	@if [ -z "$(RUN_DIR)" ]; then echo "ERROR: set RUN_DIR"; exit 1; fi

clean:
	rm -rf $(VENV)
