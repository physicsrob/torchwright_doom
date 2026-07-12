"""The model: every module that compiles into the transformer.

This package boundary is the dumb-host line.  Everything under
``model/`` is computation-graph code that ``model_graph.build_graph``
assembles and torchwright compiles into transformer weights; it runs
as attention + MLP, never as Python at render time.  Everything
outside it (``run.py``, ``config.py``, ``prompt/``, ``bundle/``,
``interpret/``, ...) runs on the host.  Nothing here imports
transformers or onnx, and the bundle's ``infer.py`` never imports
this package.

Layout: flat files at this level are the shared substrate — the
"machine" any program would need to be expressed as a transformer.
Directories are the stages of the DOOM program written in it.

Machine kernel (flat, this level):

- ``vocab`` / ``tokens`` / ``value_ranges`` / ``marker_ranges`` — the
  token vocabulary and value encodings (with ``embedding`` / ``emit``
  / ``extract`` these form one import cycle and must co-reside)
- ``embedding`` / ``emit`` / ``extract`` — token <-> residual
  encode/decode
- ``std`` — the helper-op shim lowering ported-renderer helpers onto
  torchwright ops
- ``past`` / ``attention_handles`` — reading previously-emitted
  tokens (past channels, keyed/presence/recency lookups)
- ``render_ops`` — shared forward math (atan2, distance, clamps),
  used by every write-side stage
- ``constants`` — env-driven screen sizing; ``render_constants`` —
  attention gains and protocol sentinels (disjoint concerns despite
  the sibling names)
- ``doom_lighting`` / ``asset_config`` — the data floor under the
  token contract (``vocab`` and ``value_ranges`` read them, so they
  cannot live in ``assets/``)
- ``token_match`` — the shared token-type predicate
- ``render_main`` — the assembler: ``forward()`` builds the read-side
  views, publishes protocol owners, and dispatches each branch's
  next-token.  The front door for reading the model.

Program stages (directories):

- ``scene/`` — the static read side: prefill token interpretation,
  headers, queryable map facts, the assembled SceneIndex
- ``protocol/`` — the autoregressive protocol: current-token
  interpretation and the declarative dispatch/ownership table
- ``traversal/`` — the BSP walk (R_RenderBSPNode), bbox visibility
  pruning, traversal stack edges, and the solidsegs occlusion
  channel the walk queries
- ``raster/`` — seg projection through pixels: wall ranges/columns,
  visplanes, flats, the pixel dispatcher, UV/PWL texture math, the
  weapon sprite and status bar
- ``assets/`` — WAD textures, flats, HUD and weapon graphics
  compiled to lookup tables, plus light-level colormaps

Import layering (enforced by tests/architecture/test_runtime_policy.py):
imports flow kernel -> {protocol, assets} -> scene -> traversal ->
raster; ``render_main`` is the only kernel module that imports from
the directories.  If this package ever needs further nesting, nest
exactly along these directory lines — the partition was measured
against the import graph (2026-07); every alternative cut (segs vs
pixels, read vs write) crosses real dependencies.
"""
