# Glossary

Plain-English definitions of the coined terms that recur in
`torchwright_doom`'s code and docstrings. The goal is that a reader can
understand any one docstring "cold" by looking a term up here, rather
than reverse-engineering it from use. Where a term is load-bearing in a
file, the docstring should also define it inline on first use and point
here.

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

- **marker** — a token emitted just before a carrier to say what the next
  number means (e.g. `SET_CURSOR_X` then a `VALUE`). Later positions
  recover "the value after marker M" by attending to the most recent row
  carrying M.

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

- **owner** — a module/class that publishes a sub-protocol's channels and
  builds its branch outputs (`BspTraversal`, `SegProjection`,
  `WallColumnRenderer`, …). The `forward()` dispatch routes each token
  type to its owner.

- **the `_u` suffix** — a *unified* id that is a BSP node id *or* a
  subsector id in one numbering (a BSP child is one or the other), e.g.
  `child_u` (`bsp_traversal.py`).

## DOOM renderer terms

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
