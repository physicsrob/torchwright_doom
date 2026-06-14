"""The render runtime — compile the DOOM transformer to ONNX, run it
autoregressively to *generate* a frame, decode the token stream to pixels,
and compare against the reference render.

Submodules (the import graph runs one direction, top to bottom):

- ``tokens_bridge`` — token <-> W_EMBED-row encode/decode.
- ``kv_cache`` — the runtime-owned static :class:`KVCache` + the windowed
  permanent/expiring slot-placement policy (the HF-StaticCache analog).
- ``generation`` — :class:`TokenRuntime`, the ABC every generation loop runs
  on (chunked prefill, pure AR, speculative decode as methods — the
  GenerationMixin analog).
- ``onnx_runtime`` — :class:`OnnxTokenRuntime`, the one production runtime:
  ORT session ownership, CUDA-graph-captured IO bindings, the step paths.
- ``compiled_model`` — build the forward graph + compile it to the artifact.
- ``compile_cache`` — the on-disk compile-artifact cache (key = config +
  git SHAs) + the ``OnnxDebugSession`` loader.
- ``config`` / ``wad_scene`` — YAML job config; WAD-backed scene + the in-tree
  pydoom adapter (``pydoom_scene_for``).
- ``decode`` — generated token stream -> screen pixel buffer (dumb host).
- ``compare`` — fetch the reference render + report image-level agreement stats.
- ``artifacts`` — write ``token_dump.json`` + the PNGs.
- ``diagnostic`` — chunked teacher-forced divergence localizer.
- ``profile_analysis`` — summarize ORT profiling traces (the ``--profile`` path).
- ``cli`` — argparse entrypoint over all of the above
  (``python -m torchwright_doom.inference``).

This package deliberately re-exports nothing: ``embedding`` builds the
screen-sized vocab AT IMPORT, so callers must ``apply_screen_env(config)``
before importing the modules that reach it (``tokens_bridge``,
``compiled_model``, ``compile_cache``) — the runtime trio
(``kv_cache``/``generation``/``onnx_runtime``) is import-clean by design.
The reference renderer + drafter are the in-tree ``torchwright_doom.pydoom``
package; no external checkout is required.
"""
