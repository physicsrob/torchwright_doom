# Glossary

Plain-English definitions of the coined terms that recur in
`torchwright_doom`'s code and docstrings. The goal is that a reader can
understand any one docstring "cold" by looking a term up here, rather
than reverse-engineering it from use. Where a term is load-bearing in a
file, the docstring should also define it inline on first use and point
here.

## Architecture and reading path

- **read side / write side** — the two halves of the per-token forward
  pass (`render_main.forward`). The *read side* decodes the current input
  token and consults static map facts: `vocab` / `tokens` / `embedding` /
  `extract` (token ↔ residual), `scene_*` (static map data), `protocol_*`
  (the dispatch table). The *write side* computes geometry and
  rasterization and produces the next output token: `bsp_traversal`,
  `seg_projection`, the `*_state` / `*_renderer` modules, and the pixel
  pass. Most write-side state is *published* (below) so later positions
  attend back to it.

- **the `after_<token>` convention** — an owner method `after_X()` builds
  the emit *head* (below) for the token the protocol emits *after* an `X`
  token. `render_main.build_branch_outputs` collects one per dispatch
  branch, and the output-head dispatch selects exactly one per AR step by
  the current input token's type. So `after_visit_subsector` builds what
  follows a `VISIT` token, etc. (`render_main.py` and every `*_renderer` /
  `*_state` / traversal module).

- **the import-time-node rule** — a graph `Node` (including a
  `constant(...)`) gets a global auto-incrementing id when constructed, so
  building one at *module import* time aliases it under the test harness's
  node-id reset (which expects `global_node_id == 0` after import).
  Therefore sentinel/constant nodes are built *inside* the publish
  methods, never at module scope; raw numpy/Python arrays and plain-list
  `linear` matrices may stay at module level (they are not nodes). Many
  `*_state` modules note this; the canonical statement is here.

- **Phase letters (Phase H, Phase J, …)** — port-history milestone tags
  from the original-to-graph port, kept in some module docstrings. They mark
  *when* a module was ported, not a runtime ordering: Phase H is the
  wall-column + visplane rasterization milestone (`wall_column_*`,
  `visplane_*`, `range_dispatcher`), Phase J the flat/floor-ceiling pass
  (`flat_*`, the projection texel path). Distinct from
  `SegProjection.publish`'s numbered `# Phase N` comments, which *are* its
  internal publish order (phases 1–13).

## Tokens and emission

- **token** — one discrete input/output of the autoregressive loop. Each
  token has a *type* and, for numeric types, a *payload* value. The host
  copies each output token to the next input (see `CLAUDE.md`, "Dumb host
  principle").

- **slot** — a named field of a token type with a declared range, e.g. a
  `NODE` token's `j` coordinate (`IntSlot`/`FloatSlot` in `tokens.py`).

- **carrier** — a token type whose job is to carry one wide numeric value
  as its payload (`VALUE` and `ANGLE_VALUE`). The renderer "emits a
  number" by emitting a carrier token; which number it means is set by the
  *marker* that preceded it.

- **digit-quad** — the payload encoding for a numeric value: a small block
  (2 or 4 numbers) that encodes the value's bytes — a high byte from a
  piecewise-linear (PWL) staircase plus a low byte from affine math — so a
  value spanning thousands fits a handful of residual columns
  (`emit.py`).

- **head / emit head / `head_width`** — just a token's own residual
  columns (type code + payload), *without* the shared constant tail of
  derived columns. The dispatch builds every branch at head width and
  stamps the shared tail once after picking the winner, which keeps the
  live residual narrow enough to compile (`emit.py`, `render_main.py`).

- **derived column / derived tail** — columns holding precomputed
  functions of a token (e.g. an angle's sin/cos) that are constant-zero at
  emit time. `emit_derived_zero` is the shared tail stamped after dispatch
  selection.

- **token-naming convention (provenance)** — a token's *name* advertises
  where it comes from, so a reader can tell a faithful DOOM operation from
  invented sandbox scaffolding at a glance:
  - **real DOOM functions** keep the engine's identifier verbatim, prefix
    and case intact: `R_PointOnSide`, `R_Subsector`, `R_AddLine`,
    `R_StoreWallRange`, `R_CheckPlane`, `R_MakeSpans`, `R_MapPlane`,
    `ST_Drawer`. If DOOM's C source spells it `R_*` / `ST_*`, so do we.
  - **engine nouns** — the data structures DOOM operates on — are
    lowercase, struct-style: `node`, `seg`, `drawseg`, `boxpos`, `bbox`,
    `visplane`, `pixel`.
  - **invented protocol tokens** — scaffolding with no DOOM counterpart,
    added to make the render expressible as an autoregressive stream — are
    lowerCamelCase: `noOp`, `nextSeg`, `bspFront`, `bspCheckBack`,
    `bspReturn`, `clipScan`, `segKpart`, `planeMark`, `hasBacksector`,
    `pointOnSideResult`.
  - **fields** of a record are dotted-lowercase under their owner:
    `node.x`, `seg.front.floor`, `drawseg.scale1.den`.

  This convention is **load-bearing for the readable surface's honesty
  guard** (`tokenizer/display.py`): a value-in-name relabel (folding a
  slot into the word, e.g. `hasBacksector(flag=1)` → `twoSided`) is allowed
  **only** for invented protocol tokens — never for a real DOOM call, whose
  name must stay literally what DOOM calls it. So `R_CheckPlane`'s `kind`
  renders positionally (`R_CheckPlane(floor, 0)`), it is not folded into a
  `floorCheckPlane`. A reader can therefore trust every `R_*` / `ST_*` they
  see is a real engine operation, not a prettified sandbox token.

## Type and identity matching (attention)

- **E8 code** — the fixed 8-number code in every token's embedding that
  identifies its *type*. Two tokens' types are compared by a dot-product
  of their E8 codes (`extract.type_matches`); a match scores high enough to
  resolve cleanly. (Built from `torchwright`'s spherical codes; "E8" is the
  8-dimensional code family.)

- **lifted key / lifting** — encoding an integer id as the 3-number vector
  `[id, -id², 1]` so that a dot-product with the query `[2q, 1, 1]` peaks
  exactly when `id == q` (the dot equals `1 + q² - (id-q)²`). This turns
  *equality of ids* into a single attention score, instead of a width-N
  one-hot key (`attention_handles.lifted_id_query`, `vocab._id_lifted_key`).

- **marker** — three distinct senses; the first is primary:
  1. *protocol marker* (this sense) — a token emitted just before a
     carrier to say what the next number means (e.g. `SET_CURSOR_X` then a
     `VALUE`). Later positions recover "the value after marker M" by
     attending to the most recent row carrying M.
  2. *presence-marker channel* — the `_marker` row a `RecentMarkerHandle` /
     `KeyMarkerHandle` publishes and `MARKER_PRESENT` tests, used to ask
     "did any matching producer row exist?" (`attention_handles.py`).
  3. *DOOM plane-mark* — unrelated DOOM-engine term meaning "record a
     visplane region" (`R_CheckPlane`, the `PLANE_MARK` token, the
     `VisplaneMarker` class). See **visplane**, not this entry.

- **recency / `pick_most_recent`** — attention that, among rows matching a
  key, picks the most recent one (position is the tiebreak). Long spans
  need a large `MATCH_GAIN` so a true match outscores the recency tiebreak
  (`past.py`, `render_constants.py`).

- **radix (key) / bucket / digit / successor / carry** — a trick for keys
  over a screen coordinate or id in `[0, N)`. The value is split into a
  high **bucket** digit and a low **digit** (radix = `ceil(sqrt(N))`), so a
  key that would be width-N collapses to width ~`2·sqrt(N)`. A
  **successor** scan finds the next occupied index after a given one: it is
  either in the **same** bucket at a higher digit, or — if that bucket is
  exhausted — the lowest index in the next non-empty higher bucket (a
  **carry**, by analogy with addition) (`solid_intervals.py`,
  `visplane_state.py`, `wall_column_state.py`).

## Graph state

- **publish / channel** — to *publish* a value is to compute it at every
  position and store it in the residual stream ("the past") under a name,
  so later positions can attend back to it. That stored, named,
  per-position value is a **channel** (`past.GraphPast.publish`).

- **owner** — a module/class that the `forward()` dispatch routes a set of
  token types to. Two kinds, distinguished by the file-naming convention:
  - *publishing owner* — computes and stores (*publishes*) a sub-protocol's
    channels, usually in a `publish()` method (`BspTraversal`,
    `SegProjection`, and the `*_state` modules). The `*_state.py` half of a
    pair.
  - *read-only branch owner* — reads already-published channels and builds
    one or more dispatch branches' next-token heads, publishing nothing
    (`wall_column_renderer`, `visplane_marker`, `range_dispatcher`,
    `pixel_dispatcher`, `flat_pass_renderer`, `seg_scanner`,
    `wall_range_builder`). The `*_renderer.py` / `*_marker.py` half of a
    pair.

- **handle** — a `PastHandle`: a named, reusable read capability over the
  past (one published channel), returned by `publish`. A handle's `pick`
  methods turn into the attention reads that recover that channel's value
  at a later position (`past.py`, `attention_handles.py`).

- **subcontext** — a named slice of the `SegProjection` projection record
  (`wall` / `planes` / `flats` / `seg`) that carries one downstream
  subsystem's published channels. `SegProjection.publish` runs numbered
  publish phases and hangs each subsystem's state off the matching
  subcontext, so a branch owner reads exactly the subcontext it needs
  (`seg_projection.py`).

- **validity channel** — a per-row +1/−1 marker selecting which rows a
  `mean_where`-style read averages over; rows marked invalid are excluded
  from the average (`past.py`).

- **presence lookup** — a keyed read that *also* answers whether any active
  producer matched, instead of every key gating to zero on a miss. Distinct
  from a plain **value lookup**, which assumes the key exists and returns
  its field (`scene_facts.py`, `attention_handles.py`).

- **the `_u` suffix** — a *unified* id that is a BSP node id *or* a
  subsector id in one numbering (a BSP child is one or the other), e.g.
  `child_u` (`bsp_traversal.py`).

## Static scene read side

- **scene fact / scene index** — static map data (vertices, segs,
  subsectors, BSP nodes, sector/plane info) published once from the prefill
  so the render code can attend back to it for the rest of the frame. The
  read API is `scene_index.SceneIndex` (the bundle) over
  `scene_facts.py` (the individual keyed lookups).

- **header context** — the most-recent structural header row (a `NODE` /
  `SUBSECTOR` / `SEG` / `PLANE_DEF` token) that establishes *which* record
  the following marker/value tokens belong to. Recovered by recency
  attention, so a `VALUE` after a `NODE` header is read as that node's
  field (`scene_headers.HeaderContext`).

## DOOM renderer terms

- **span** — a horizontal run of floor/ceiling pixels at one screen row
  inside a single visplane (DOOM's `ds_x1..ds_x2` in `R_MakeSpans` /
  `R_MapPlane`). The flat pass walks each visplane's columns, opens/closes
  spans as the top/bottom coverage changes between adjacent columns, then
  fills each open span left-to-right (`flat_state.py`, `flat_*`).

- **flat** — DOOM's name for a floor/ceiling texture (as opposed to a
  *wall* texture). The "flat pass" fills floor/ceiling pixels.

- **visplane** — DOOM's per-frame record of one floor/ceiling plane to
  fill: a screen region at a single height, light level, and flat.

- **drawseg** — DOOM's per-visible-wall-segment record (perspective scale,
  texture vertical origins, sprite-clip silhouette heights) produced by
  `R_StoreWallRange`.

- **DOOM C provenance** — comments like `# DOOM: R_RenderBSPNode (r_bsp.c)`
  point at the original engine function this code mirrors. These are kept
  as durable cross-references.

## Assets and lighting

Textures and flats are compiled to lookup tables at import time (plain
numpy, never graph nodes — see *the import-time-node rule*), then read in
the graph by `table_lookup_2d`.

- **bank** — a group of wall textures sharing one `(width, height)` table
  shape, so they can index one 3-D `table[local_id, v, u]` (`WallBank` in
  `asset_banks.py`). `bank_id` identifies the shape-group.

- **sawtooth bank** — one `v mod H` sawtooth piecewise-linear curve per
  *distinct* wall-texture height `H`, used to wrap a wall's vertical texture
  coordinate (`pwl_banks.build_sawtooth_bank`, `SAWTOOTH_BANK`).

- **`h_idx_oh`** — a per-texture one-hot over the distinct heights in
  `WALL_HEIGHT_BANK`, selecting which height's sawtooth applies to a given
  wall span (`asset_banks.py`, `pwl_banks.py`).

- **COLORMAP** — DOOM's light-shading indirection: a lighting computation
  yields a *colormap row*, and that row plus the texture's palette index
  gives the final palette index. PLAYPAL (the 256-colour palette) is
  applied last. The lighting code returns the colormap row, not RGB
  (`doom_lighting.py`).

- **light level / light-num** — the per-sector brightness index (`0..15`,
  `LIGHTLEVELS=16`), adjusted by wall orientation (DOOM makes horizontal
  walls one level darker, vertical ones one level lighter). It indexes the
  distance-shading tables (`doom_lighting.py`).

- **scalelight / zlight** — DOOM's two distance→colormap-row tables:
  `scalelight` for textured walls (indexed by perspective scale),
  `zlight` for flats (indexed by view distance). `startmap` is the
  starting colormap row a given light level maps to before the distance
  term (`doom_lighting.py`).

## Prefill and scene construction

- **prefill** — the token sequence describing the static scene and the
  initial player pose that the model reads *before* autoregression begins
  (`prompt/build.build_prompt`). Analogous to the prompt an LLM ingests
  before it starts generating.

- **subset** — bbox-slicing a full WAD map down to the segs / subsectors /
  minimal BSP subtree that the frame needs, renumbered to dense indices so
  ids stay small (`prompt/subset.subset_by_bbox`).

- **mean-centred / `scene_origin`** — the subset step shifts every kept
  coordinate by the centroid of the kept vertices so values fit the
  `VALUE` ranges; the centroid is stored in `scene_origin` and the player
  pose is shifted by it to stay in the same frame (`prompt/types.py`,
  `prompt/subset.py`).

## Runtime

The production runtime (`inference/`) drives the compiled transformer
through the same autoregressive loop any chat model uses; these terms name
its host-side machinery.

- **windowed KV cache** — a fixed-size key/value cache (`model.cache_window`
  slots). Every committed row is either *permanent* (resident for the whole
  run) or *expiring* (its slot may be recycled once the window fills),
  decided purely by the row's token type (`kv_cache.py`; the certified
  expiring set and its invariant are in `CLAUDE.md`, "Windowed KV cache").

- **recycle** — reuse of an expiring row's cache slot once the window
  fills. `alloc_slot` refuses to recycle a slot younger than the worst
  certified read scope, so a still-needed expiring row is never evicted
  (`kv_cache.min_recycle_distance_for`).

- **writtenness mask** — the in-graph signal of which cache rows hold real
  committed values vs unwritten slots, so the host's slot placement stays
  invisible to the graph math (`compiled_model.py`, `kv_cache.py`).

- **speculative decode / draft / reuse buffer** — the spec-decode loop
  runs a cheap draft pass to propose several next tokens, then verifies
  them in one batched model pass and keeps the accepted prefix. A partially
  rejected draft tail is often still correct one step later, so it is held
  in a *reuse buffer* and re-offered (`generation.spec_decode_rollout`).
