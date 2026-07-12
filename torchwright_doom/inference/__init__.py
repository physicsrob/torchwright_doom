"""Direct-HF compilation, portable stock-Phi-3 inference, and diagnostics.

Submodules (the import graph runs one direction, top to bottom):

- ``tokens_bridge`` — token <-> W_EMBED-row encode/decode.
- ``hf_bundle`` — compile a complete stock ``Phi3ForCausalLM`` bundle directly
  from ``build_graph``, validate it, and publish it atomically.
- ``../infer.py`` — the isolated ``model.generate`` CLI copied to each
  bundle's root and executed verbatim by production rendering.
- ``compiled_model`` — the sole source graph builder plus explicit ONNX debug
  compilation.
- ``compile_cache`` — diagnostic-only ONNX cache and ``OnnxDebugSession`` loader.
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
``compiled_model``, ``compile_cache``, ``hf_bundle``). The reference renderer +
token state machine are the in-tree ``torchwright_doom.pydoom`` package; no
external checkout is required.
"""
