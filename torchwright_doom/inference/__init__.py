"""The render runtime — compile the DOOM transformer to ONNX, run it
autoregressively to *generate* a frame, decode the token stream to pixels,
and compare against the reference render.

Submodules (the import graph runs one direction, top to bottom):

- ``tokens_bridge`` — token <-> W_EMBED-row encode/decode.
- ``kv_cache`` — the runtime-owned unbounded :class:`KVCache` (the type the
  generation-loop signatures name; the HF-DynamicCache analog).
- ``generation`` — :class:`TokenRuntime`, the ABC every generation loop runs
  on (chunked prefill, pure AR as methods — the GenerationMixin analog).
- ``hf_runtime`` — :class:`HfTokenRuntime`, the production runtime: a native
  HuggingFace ``TorchwrightForCausalLM`` (converted from the ONNX artifact)
  driven over the row-id seam.
- ``hf_export`` — convert the artifact to a trust-remote-code HF bundle (the
  Hub publish path). The render + pydoom gate is ``cli.run_config`` itself
  (``make run COMPARE=1``).
- ``compiled_model`` — build the forward graph + compile it to the artifact.
- ``compile_cache`` — the on-disk compile-artifact cache (key = config +
  git SHAs) + the ``OnnxDebugSession`` loader.
- ``config`` / ``wad_scene`` — YAML job config; WAD-backed scene + the in-tree
  pydoom adapter (``pydoom_scene_for``).
- ``decode`` — generated token stream -> screen pixel buffer (dumb host).
- ``compare`` — fetch the reference render + report image-level agreement stats.
- ``artifacts`` — write ``token_dump.json`` + the PNGs.
- ``diagnostic`` — chunked teacher-forced divergence localizer.
- ``cli`` — argparse entrypoint over all of the above
  (``python -m torchwright_doom.inference``).

This package deliberately re-exports nothing: ``embedding`` builds the
screen-sized vocab AT IMPORT, so callers must ``apply_screen_env(config)``
before importing the modules that reach it (``tokens_bridge``,
``compiled_model``, ``compile_cache``, ``hf_runtime``) — ``kv_cache`` /
``generation`` are import-clean by design.  The reference renderer + token
state machine are the in-tree ``torchwright_doom.pydoom`` package; no external
checkout is required.
"""
