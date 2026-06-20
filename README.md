# torchwright_doom

DOOM rendering and game graph compilation, built on [torchwright](https://github.com/physicsrob/torchwright). Flagship project of the [torchdoom](https://github.com/physicsrob/torchdoom) umbrella.

## What this is

This package builds a **computation graph** that the `torchwright` compiler
turns into a **transformer**, which then renders DOOM **autoregressively** —
one discrete input token in, one output token out, per step, the same loop
that drives any chat model. Output tokens carry pixel information; the host
copies each output token to the next input and blits pixels to the screen.

The **dumb-host principle** governs everything: all rendering logic — wall
selection, visibility, distance, texture lookup, compositing — lives inside
the transformer. The host only feeds tokens and writes pixels; it does no
geometry, sorting, or arithmetic. (See `CLAUDE.md`.)

## Where to start

- **Entry point:** the per-token forward pass is `render_main.forward`
  (`torchwright_doom/render_main.py`). It builds the read side (decode the
  input token + consult static map facts), publishes the write-side
  protocol owners, builds each dispatch branch's next-token, and selects one
  by the current token's type.
- **Reading path** (one `forward()` pass, read side → write side):
  `vocab` / `tokens` → `embedding` / `extract` → `scene_tokens` /
  `scene_headers` / `scene_index` / `scene_facts` (static read side) →
  `protocol_tokens` / `protocol_registry` (the dispatch table) →
  `render_main.forward` (assembly) → the write side: `bsp_traversal`
  (R_RenderBSPNode) → `seg_projection` → the `wall_*` / `visplane_*` /
  `flat_*` rasterizers → the pixel pass.
- **Prefill pipeline** (WAD → tokens the model reads before autoregression):
  `doom1.wad` → `prompt/wad.py` (raw `MapData`) → `prompt/subset.py`
  (bbox-sliced, renumbered, mean-centred) → `prompt/build.py` (`list[Token]`)
  → `inference/tokens_bridge.py` (row indices) → the model. Production entry:
  `inference/wad_scene.prefill_rows_for`.

## Docs

- **`CLAUDE.md`** — full module layout, the windowed-KV-cache invariant, and
  the graph-debugging tool sequence.
- **`GLOSSARY.md`** — plain-English definitions of the coined vocabulary
  (carrier, head, marker, owner, subcontext, visplane, flat, …).
- **`PROTOCOL.md`** — the pixel protocol: the exact per-frame token
  sequence (prefill + every AR phase), in the readable-surface token names.
- **`protocol_registry.render_protocol_table()`** — the generated table of
  the token protocol (every token type, its phase, role, and dispatch
  wiring), for a top-down view of the AR protocol.

## Running

There is exactly one committed render config, `configs/e1m1.yaml`. Use
`make run` (renders on Modal) and `make help` for the menu.
