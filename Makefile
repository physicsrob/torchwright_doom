# THE production config. The only other maintained YAML is the low-resolution
# validation config; experiments copy one to /tmp and override CONFIG=.
#
# Render-job defaults (pose and max new tokens) live in the
# config's `run:` section — NOT here.  The Makefile passes a flag only when
# the variable is set explicitly (e.g. `make run MAX_NEW_TOKENS=8000`),
# so it can never hold a stale copy of a default.
CONFIG ?= configs/e1m1.yaml
OUT_DIR ?= out/render
# Modal GPU for render_remote (read at modal_render.py import as an env
# var). B200 is the default: the dense checkpoint is ~98 GB fp32 plus a growing
# generation cache, and B200's 192 GB HBM fits it.
RENDER_GPU ?= b200
PNG_ZOOM ?= 8

# Verbose compile is ON by default (streams the compiler's per-layer detail +
# head-pruning summary to stdout); opt out with VERBOSE_COMPILE=0.
VERBOSE_COMPILE ?= 1
_RENDER_VERBOSE_COMPILE := $(if $(filter-out 0,$(VERBOSE_COMPILE)),--verbose-compile)
# DISABLE_CACHE=1 make compile — production compile that neither reads nor
# writes the durable caches (complete HF_BUNDLE_VOLUME + SCHEDULE_VOLUME);
# the sampled schedule is saved to local /tmp instead (path printed).  Any
# non-empty, non-0 value enables it.  `compile` only.
_RENDER_DISABLE_CACHE := $(if $(filter-out 0,$(DISABLE_CACHE)),--disable-cache)
_RENDER_PNG := $(if $(PNG),--png)
_RENDER_COMPARE := $(if $(COMPARE),--compare)
_RENDER_COMPILE_ARGS = $(strip \
	--config $(CONFIG) \
	$(_RENDER_VERBOSE_COMPILE) \
	$(_RENDER_DISABLE_CACHE) \
)
_RENDER_RUN_ARGS = $(strip \
	--config $(CONFIG) \
	$(if $(RENDER_X),--x $(RENDER_X)) \
	$(if $(RENDER_Y),--y $(RENDER_Y)) \
	$(if $(RENDER_ANGLE),--angle $(RENDER_ANGLE)) \
	$(if $(RENDER_VIEWZ),--viewz $(RENDER_VIEWZ)) \
	--out-dir $(OUT_DIR) \
	$(if $(MAX_NEW_TOKENS),--max-new-tokens $(MAX_NEW_TOKENS)) \
	--png-zoom $(PNG_ZOOM) \
	$(_RENDER_PNG) \
	$(_RENDER_COMPARE) \
	$(_RENDER_VERBOSE_COMPILE) \
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
# durable HF_BUNDLE_VOLUME (not local disk); a later `make run` is a cache hit.
# The log file is timestamp+pid-unique so parallel invocations never share
# one; the /tmp/torchwright_doom-compile.log symlink is last-wins.
render-compile compile:
	@bash -c ' \
		LOGFILE=/tmp/torchwright_doom-compile-$$(date +%Y%m%d-%H%M%S)-$$$$.log ; \
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

.PHONY: compile-onnx-debug
compile-onnx-debug:
	uv run python -m torchwright_doom compile-onnx-debug \
		--config $(CONFIG) $(_RENDER_VERBOSE_COMPILE)

.PHONY: probe-volume-publication
probe-volume-publication:
	uv run modal run modal_render.py::probe_volume_publication

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

# The production correctness gate is `make run COMPARE=1` (it scores the HF
# render's coverage / within-option color against the pydoom reference and
# writes the diff PNG). ~30 min/frame on the render GPU; too heavy for
# per-commit `make test`, run manually.

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
