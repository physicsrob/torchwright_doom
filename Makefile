CONFIG ?= configs/e1m1_start_room.yaml
OUT_DIR ?= out/render
RENDER_X ?= 1056.0
RENDER_Y ?= -3616.0
RENDER_ANGLE ?= 64
RENDER_VIEWZ ?= 41.0
RENDER_MODE ?= spec_decode
# 10240 covers the full e1m1 frame (9739 rollout tokens to DONE) + margin.
# Smaller gate configs (e1m1_l4_d3072, S=4608) must pass an explicit
# lower value or empty_past() rejects the demand.
RENDER_MAX_POSITIONS ?= 10240
# Modal GPU for render_remote (read at modal_render.py import).
# a100-80gb | b200 — the captured decode is bandwidth-bound, so B200's
# ~4x HBM bandwidth maps ~directly to step time.
RENDER_GPU ?= a100-80gb
RENDER_DRAFT_WINDOW ?= 8
# 1024-row chunks bound the static-S prefill logits transient
# ((n_heads, chunk, S) per layer; unchunked at n=3613, S=12288 the widest
# layer is ~45 GB on the A100 and OOMs the L4 gate).  Chunking is
# semantically identical — see plan_cuda_graph_decode.md "Memory budget".
PREFILL_CHUNK_SIZE ?= 1024
RENDER_PROGRESS_EVERY ?= 250
PNG_ZOOM ?= 8

_RENDER_VERBOSE_COMPILE := $(if $(VERBOSE_COMPILE),--verbose-compile)
_RENDER_PNG := $(if $(PNG),--png)
_RENDER_COMPARE := $(if $(COMPARE),--compare)
_RENDER_PROFILE := $(if $(PROFILE),--profile)
_RENDER_COMPILE_ARGS = $(strip \
	--config $(CONFIG) \
	$(_RENDER_VERBOSE_COMPILE) \
)
_RENDER_RUN_ARGS = $(strip \
	--config $(CONFIG) \
	--x $(RENDER_X) \
	--y $(RENDER_Y) \
	--angle $(RENDER_ANGLE) \
	--viewz $(RENDER_VIEWZ) \
	--mode $(RENDER_MODE) \
	--out-dir $(OUT_DIR) \
	--max-positions $(RENDER_MAX_POSITIONS) \
	--draft-window $(RENDER_DRAFT_WINDOW) \
	--prefill-chunk-size $(PREFILL_CHUNK_SIZE) \
	--progress-every $(RENDER_PROGRESS_EVERY) \
	--png-zoom $(PNG_ZOOM) \
	$(_RENDER_PNG) \
	$(_RENDER_COMPARE) \
	$(_RENDER_VERBOSE_COMPILE) \
	$(_RENDER_PROFILE) \
)
_RENDER_MODAL_ARGS = $(strip \
	$(_RENDER_RUN_ARGS) \
	$(if $(RUN_NAME),--run-name $(RUN_NAME)) \
)

.PHONY: lint
lint:
	uv run black --check .
	uv run mypy .

.PHONY: render-compile compile
render-compile compile:
	uv run python -m torchwright_doom.render compile $(_RENDER_COMPILE_ARGS)

.PHONY: render-run run
render-run run:
	@bash -c ' \
		LOGFILE=/tmp/torchwright_doom-render-run-$$(date +%Y%m%d-%H%M%S).log ; \
		ln -sfn "$$LOGFILE" /tmp/torchwright_doom-render-run.log ; \
		echo "=== Log file: $$LOGFILE ===" | tee "$$LOGFILE" ; \
		echo "=== Running render on Modal ===" | tee -a "$$LOGFILE" ; \
		start=$$(date +%s) ; \
		RENDER_GPU=$(RENDER_GPU) uv run modal run modal_render.py $(_RENDER_MODAL_ARGS) \
			2>&1 | tee -a "$$LOGFILE" ; \
		rc=$${PIPESTATUS[0]} ; \
		end=$$(date +%s) ; \
		echo "" | tee -a "$$LOGFILE" ; \
		echo "=== Render finished in $$((end - start))s (exit $$rc) ===" | tee -a "$$LOGFILE" ; \
		echo "=== Log file: $$LOGFILE ===" | tee -a "$$LOGFILE" ; \
		exit $$rc \
	'

.PHONY: render-run-local run-local
render-run-local run-local:
	uv run python -m torchwright_doom.render run $(_RENDER_RUN_ARGS)

.PHONY: test
test:
	@bash -c ' \
		LOGFILE=/tmp/torchwright_doom-test-$$(date +%Y%m%d-%H%M%S).log ; \
		ln -sfn "$$LOGFILE" /tmp/torchwright_doom-test.log ; \
		echo "=== Log file: $$LOGFILE ===" | tee "$$LOGFILE" ; \
		echo "=== Running tests on Modal ===" | tee -a "$$LOGFILE" ; \
		echo "=== Monitor: make test-logs ===" | tee -a "$$LOGFILE" ; \
		start=$$(date +%s) ; \
		uv run modal run modal_test.py \
			--file $(if $(FILE),$(FILE),tests) \
			$(if $(ARGS),--args "$(ARGS)") \
			2>&1 | tee -a "$$LOGFILE" ; \
		rc=$${PIPESTATUS[0]} ; \
		end=$$(date +%s) ; \
		echo "" | tee -a "$$LOGFILE" ; \
		echo "=== Tests finished in $$((end - start))s (exit $$rc) ===" | tee -a "$$LOGFILE" ; \
		echo "=== Log file: $$LOGFILE ===" | tee -a "$$LOGFILE" ; \
		exit $$rc \
	'

.PHONY: test-logs
test-logs:
	@tail -f /tmp/torchwright_doom-test.log

.PHONY: test-local
test-local:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE=<path> is required for test-local." >&2 ; \
		echo "       test-local runs pytest on the local machine and must target" >&2 ; \
		echo "       a single file to avoid accidentally running the whole suite" >&2 ; \
		echo "       (which belongs on Modal via 'make test')." >&2 ; \
		echo "Example: make test-local FILE=tests/path/to_test.py" >&2 ; \
		exit 2 ; \
	fi
	uv run pytest $(FILE) $(ARGS)

.PHONY: modal-run
modal-run:
	@if [ -z "$(MODULE)$(SCRIPT)" ]; then \
	    echo "Error: MODULE=<dotted.name> or SCRIPT=<path> required." >&2 ; \
	    echo "Example: make modal-run MODULE=<some.committed.module>" >&2 ; \
	    exit 2 ; \
	fi
	@bash -c ' \
		LOGFILE=/tmp/torchwright_doom-modal-run-$$(date +%Y%m%d-%H%M%S).log ; \
		ln -sfn "$$LOGFILE" /tmp/torchwright_doom-modal-run.log ; \
		echo "=== Log file: $$LOGFILE ===" | tee "$$LOGFILE" ; \
		echo "=== Running on Modal ===" | tee -a "$$LOGFILE" ; \
		start=$$(date +%s) ; \
		uv run modal run modal_run.py \
		    $(if $(MODULE),--module $(MODULE)) \
		    $(if $(SCRIPT),--script $(SCRIPT)) \
		    $(if $(ARGS),--args "$(ARGS)") \
		    $(if $(CPU_ONLY),--cpu-only) \
		    2>&1 | tee -a "$$LOGFILE" ; \
		rc=$${PIPESTATUS[0]} ; \
		end=$$(date +%s) ; \
		echo "" | tee -a "$$LOGFILE" ; \
		echo "=== Finished in $$((end - start))s (exit $$rc) ===" | tee -a "$$LOGFILE" ; \
		echo "=== Log file: $$LOGFILE ===" | tee -a "$$LOGFILE" ; \
		exit $$rc \
	'
