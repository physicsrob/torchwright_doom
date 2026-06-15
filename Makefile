# THE config — the single committed configuration (see CLAUDE.md, "One
# configuration").  Experiments copy it to /tmp and override CONFIG=.
#
# Render-job defaults (pose, mode, max positions, draft window, prefill
# chunk) live in the config's `run:` section — NOT here.  The Makefile
# passes a flag only when the variable is set explicitly
# (e.g. `make run RENDER_MODE=pure_ar`), so it can never hold a stale
# copy of a default.
CONFIG ?= configs/e1m1.yaml
OUT_DIR ?= out/render
# Modal GPU for render_remote (read at modal_render.py import as an env
# var).  b200 | a100-80gb — B200 is the default: the captured decode is
# bandwidth-bound (B200's ~4x HBM bandwidth maps ~directly to step time,
# measured 14.3 ms + 0.522 us/slot x S_eff per width-1 step).  With the
# windowed cache (~11.4 GB KV + ~17 GB weights) the A100 also has slack.
RENDER_GPU ?= b200
RENDER_PROGRESS_EVERY ?= 250
PNG_ZOOM ?= 8
# Attention-window bucket table (stride bucketing): comma-separated S_eff
# list, e.g. RENDER_ATTENTION_BUCKETS=16384,32768,49152,65536.  A RUNTIME
# knob (no recompile); empty = quarters of the model's cache_stride.
RENDER_ATTENTION_BUCKETS ?=

_RENDER_VERBOSE_COMPILE := $(if $(VERBOSE_COMPILE),--verbose-compile)
_RENDER_PNG := $(if $(PNG),--png)
_RENDER_COMPARE := $(if $(COMPARE),--compare)
_RENDER_PROFILE := $(if $(PROFILE),--profile)
_RENDER_BUCKETS := $(if $(RENDER_ATTENTION_BUCKETS),--attention-buckets $(RENDER_ATTENTION_BUCKETS))
_RENDER_COMPILE_ARGS = $(strip \
	--config $(CONFIG) \
	$(_RENDER_VERBOSE_COMPILE) \
)
_RENDER_RUN_ARGS = $(strip \
	--config $(CONFIG) \
	$(if $(RENDER_X),--x $(RENDER_X)) \
	$(if $(RENDER_Y),--y $(RENDER_Y)) \
	$(if $(RENDER_ANGLE),--angle $(RENDER_ANGLE)) \
	$(if $(RENDER_VIEWZ),--viewz $(RENDER_VIEWZ)) \
	$(if $(RENDER_MODE),--mode $(RENDER_MODE)) \
	--out-dir $(OUT_DIR) \
	$(if $(RENDER_MAX_POSITIONS),--max-positions $(RENDER_MAX_POSITIONS)) \
	$(if $(RENDER_DRAFT_WINDOW),--draft-window $(RENDER_DRAFT_WINDOW)) \
	$(if $(PREFILL_CHUNK_SIZE),--prefill-chunk-size $(PREFILL_CHUNK_SIZE)) \
	--progress-every $(RENDER_PROGRESS_EVERY) \
	--png-zoom $(PNG_ZOOM) \
	$(_RENDER_PNG) \
	$(_RENDER_COMPARE) \
	$(_RENDER_VERBOSE_COMPILE) \
	$(_RENDER_PROFILE) \
	$(_RENDER_BUCKETS) \
)
_RENDER_MODAL_ARGS = $(strip \
	$(_RENDER_RUN_ARGS) \
	$(if $(RUN_NAME),--run-name $(RUN_NAME)) \
)

.PHONY: lint
lint:
	uv run black --check .
	uv run mypy .
	uv run ruff check --select F .

.PHONY: render-compile compile
# Compile on Modal — the SAME 64-CPU compile_remote container `make run` uses
# on a cache miss, so the wide CP-SAT search finds a better (fewer-layer)
# schedule than a local box can in the time budget. The artifact lands in the
# durable CACHE_VOLUME (not local disk); a later `make run` is a cache hit.
# (Local compile still happens implicitly via `make run-local` on a miss.)
render-compile compile:
	@bash -c ' \
		LOGFILE=/tmp/torchwright_doom-compile-$$(date +%Y%m%d-%H%M%S).log ; \
		ln -sfn "$$LOGFILE" /tmp/torchwright_doom-compile.log ; \
		echo "=== Log file: $$LOGFILE ===" | tee "$$LOGFILE" ; \
		echo "=== Compiling on Modal (64-CPU CP-SAT) ===" | tee -a "$$LOGFILE" ; \
		start=$$(date +%s) ; \
		uv run modal run modal_render.py::compile_only $(_RENDER_COMPILE_ARGS) \
			2>&1 | tee -a "$$LOGFILE" ; \
		rc=$${PIPESTATUS[0]} ; \
		end=$$(date +%s) ; \
		echo "" | tee -a "$$LOGFILE" ; \
		echo "=== Compile finished in $$((end - start))s (exit $$rc) ===" | tee -a "$$LOGFILE" ; \
		echo "=== Log file: $$LOGFILE ===" | tee -a "$$LOGFILE" ; \
		exit $$rc \
	'

.PHONY: render-run run
render-run run:
	@bash -c ' \
		LOGFILE=/tmp/torchwright_doom-render-run-$$(date +%Y%m%d-%H%M%S).log ; \
		ln -sfn "$$LOGFILE" /tmp/torchwright_doom-render-run.log ; \
		echo "=== Log file: $$LOGFILE ===" | tee "$$LOGFILE" ; \
		echo "=== Running render on Modal ===" | tee -a "$$LOGFILE" ; \
		start=$$(date +%s) ; \
		RENDER_GPU=$(RENDER_GPU) uv run modal run modal_render.py::main $(_RENDER_MODAL_ARGS) \
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
	uv run python -m torchwright_doom.inference run $(_RENDER_RUN_ARGS)

.PHONY: test
test: lint
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
