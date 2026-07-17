# Pixel Protocol

The autoregressive (AR) control-and-render protocol for the DOOM
renderer: the exact sequence of tokens the transformer emits to draw one
frame, one token per step, the host copying each output token to the next
input (see `CLAUDE.md`, "Dumb host principle").

This document is the protocol's *narrative* companion to two generated
sources of truth, and is kept in step with them:

- **`protocol_registry.render_protocol_table()`** — every token type, its
  phase, owner, role, and dispatch wiring (the *static* table).
- **`pydoom/drafter.py`** — the reference state machine that predicts the
  exact token order (the *dynamic* sequence): the test-only conformance
  oracle the rollout is diffed against in the graph gates (see *drafter*
  in `GLOSSARY.md`). It never runs during inference — at inference the
  transformer alone produces tokens. When this prose and the drafter
  disagree, the drafter is right.

Token names below are the **readable-surface** names emitted by
`tokenizer/surface.py` — the same names the blog's "this is the real
prompt" figure and the HF tokenizer use. They are the token types'
canonical `.name`s (`vocab.py`), not the Python constants
(`R_AddLine`, not `PROCESS_SEG`).

## Notation: the readable surface

Each token renders in functional style — `TYPE(arg, ...)`, or bare `TYPE`
when it has no args. Integer slots render `name=value`
(`R_Subsector(s=5, depth=3)`) unless the type's name already says what each
value is, in which case they go **positional** (`R_AddLine(34)`,
`R_CheckPlane(floor, 0)`, `pixel(143, 1)`). The labels are pretty by default
(reversible bijections, so the id stream is unchanged): names are **aliased**
(the `node.` / `seg.` / `front.` prefixes drop, `SSECTOR`→`ss`,
`ceiling`→`ceil`); opaque integer slots **decode** to their source word
(`child1(ss5)`, `wall_kind=portal`, booleans as `no`/`yes`, the bbox region as a
2-letter grid code); and a few **sandbox** tokens fold a dominant slot into the
word itself (`oneSided`/`twoSided`, `floorMark`/`ceilMark`). The honesty guard
keeps real DOOM calls (`R_*` / `ST_*`) literal — `R_CheckPlane`'s `kind` goes
positional, never into the name. The full label rules live in
`tokenizer/display.py` (and, in the torchdoom umbrella checkout, the blog's
`pieces/doom/docs/surface_legend.md`); the rendered ground truth is the two
figures.

**Carriers fold into their marker.** The stream carries wide numbers as a
two-token pair: a **marker** token immediately followed by its **carrier**
— a `value` token (a float squashed into the marker's `[-1, 1]` range) or
an `angleValue` token (a signed BAM integer). The marker chooses the range
(`marker_ranges.MARKER_RANGE`); the carrier never stores it. The `R`-tags
in comments below (`R0`, `R3`–`R9`) name those per-marker numeric ranges —
their bounds live in `marker_ranges.py`. In the
readable surface the carrier folds into the marker's parens as a trailing
positional, so this doc writes the *de-quantized physical value* in angle
brackets there:

    angle1(<world angle to endpoint A>)      # marker `angle1` + an angleValue
    segDcTmidMid(<middle dc_texturemid>)     # marker `segDcTmidMid` + a value
    wallColU(u_idx=<col>, <encoded scale>)   # int slot + a folded value carrier

Underneath, every such line is still **two token ids** (the marker, then
the `value`/`angleValue`). A marker like `setCursorY` carries a value in
some contexts and not others; the surface decides by lookahead (is the
next token a carrier?), so the same marker reads both ways.

**Lines.** Each "header" token type starts a new display line; following
tokens join it with spaces (or, under the `indent` knob, pack into aligned
columns). Layout is pure whitespace — it never changes the id stream.

The protocol procedures below are written one conceptual step per line.
`<computed>` names a value the graph derives; its implementation lives in
the owner modules, not here.

## Prefill (the prompt)

Before autoregression the model reads a flat prompt encoding the sliced
scene (`prompt/build.py`). It is replayed verbatim, not predicted.

**What the prompt may contain:** any **view-independent** static-scene
fact — raw map records, and static per-seg classifications derived from
them host-side before generation (e.g. `emptyLine`, `closedDoor`,
`lightStatic` below, DOOM's `R_AddLine`-time facts about the *map*, not
the *view*). All **view-dependent** work — visibility, ordering,
projection, occlusion — happens inside the transformer. This is the
input-side dumb-host boundary (see README).

Block order:

```
viewx(<x>) viewy(<y>) viewz(<z>) viewangle(<bam>)         # player state

for each BSP node j:
    node(j=j)
    x(<px>) y(<py>) dx(<dx>) dy(<dy>)                     # partition line
    child1(<front child>) child0(<back child>)            # unified node/ss ids
    bbox1.top/bottom/left/right(<edge>)                   # front child bbox
    bbox0.top/bottom/left/right(<edge>)                   # back child bbox

for each subsector s:
    ss(s=s)                                              # SSECTOR, aliased
    for each seg i in s:
        seg(i=i, is_first_of_ss=<no|yes>)
        v1.x(<ax>) v1.y(<ay>) v2.x(<bx>) v2.y(<by>)
        <oneSided|twoSided> normalAngle(<bam>)            # hasBacksector folded
        floor(<h>) ceil(<h>)                              # seg.front.floor / .ceiling
        back.floor(<h|none>) back.ceil(<h|none>)          # `none` = one-sided sentinel
        texture.mid/upper/lower(<wad name|none>)
        lightStatic(light=<l>) emptyLine(<no|yes>) closedDoor(<no|yes>)
        pegging(<no|yes>, <no|yes>) rowoffset(<o>)        # dontpegtop, dontpegbottom

for each used visplane p:                                 # floor/ceiling plane defs
    planeDef(p=p, flat_id=<flat>)
    planeHeight(<h>) planeLight(light=<l>)
for each subsector s with planes:
    ssFloorPlane(s=s, p=<p>) ssCeilingPlane(s=s, p=<p>)

begin                                                     # final prefill marker; seeds AR
```

The render of E1M1's real prefill ships in the torchdoom umbrella checkout
as the blog's `pieces/doom/e1m1_prompt_surface.txt`.

## AR entry

Autoregression begins from `begin` (the last prefill token). The first
emitted token sets the shared cursor's vertical advance for the wall pass:

```
[begin]
setCursorDirectionY        # pixels advance downward (the wall-pass direction)
```

The flat pass later flips this to `setCursorDirectionX`; the weapon pass
flips it back.

## Side-bit setup pass

Before walking the tree, record which side of every BSP node's partition
line the viewer is on. A two-token call/result per node, looping over node
ids until the prefill runs out of nodes.

```
procedure side_bit_setup_pass():
    j = 0
    while node_exists(j):
        R_PointOnSide(node=j)
        <frontSideResult|backSideResult>(node=j)        # pointOnSideResult, side folded
        j = j + 1
```

## BSP traversal

The BSP tree is walked depth-first from the root. Each internal node
descends its front child unconditionally, then its back child only if the
node's bounding box has an uncovered screen column. "Front"/"back" here
are view-dependent: the walk tests which side of the node's static
partition line the camera is on (`R_PointOnSide` — the side-bit setup
pass above) — the prefill's `child1`/`child0` record the static children,
and the side bit decides which one is "front" for this camera.

```
procedure visit_node(n, d):
    bspFront(node=n, depth=d)
    visit_child(<front child of n>, d+1)

    bspCheckBack(node=n, depth=d)
    if bbox_check(n, d):              # see "Back-child bbox check"
        visit_child(<back child of n>, d+1)

    bspReturn(entity_u=node<n>, depth=d)  # unified id, decoded (node<n> or ss<s>)

procedure visit_subsector(s, d):
    R_Subsector(s=s, depth=d)
    for each seg i in subsector s, in order:
        R_AddLine(i)
        seg_projection(i)            # see "Seg projection"; may close with nextSeg(i)
    bspReturn(entity_u=ss<s>, depth=d)    # leaf: N_NODES_MAX + s, decoded to ss<s>
```

`R_AddLine` is scoped by the most recent `R_Subsector`; the depth and
subsector id are recovered from that row, not repeated on each seg.
`bspReturn` carries the unified entity id (a node id, or
`N_NODES_MAX + s` for a subsector leaf).

`nextSeg(i)` is a per-seg *dead-end* closer, not a uniform separator: the
drafter emits it when a seg contributes no wall run to the end of its
projected range — it projects to no columns, or its run scan steps past
the last column through already-covered space. Segs skipped before
projection, and segs whose runs reach the end of the range, are followed
directly by the next `R_AddLine` (or the leaf's `bspReturn`) with no
`nextSeg`. The exact firing is the run-scan terminator; the authority is
`_SegState.completion` in `pydoom/drafter.py`.

## Back-child bbox check

Invoked after `bspCheckBack`. It can short-circuit three ways: the boxpos
fast-pass (box straddles the view center → always visible), an empty
projected range (→ skip the subtree), or a screen scan that finds an
uncovered column.

```
procedure bbox_check(n, d) -> bool:
    boxpos(<region grid code>)                  # XY: column L|C|R, row A|C|B
    if the boxpos is the center cell:           # fast-pass: trivially visible
        return true

    bx1(<region>, <first corner x>) by1(<region>, <first corner y>)
    bangle1(<first corner world angle>) btheta1(<first corner view angle>)
    bx2(<region>, <second corner x>) by2(<region>, <second corner y>)
    bangle2(<second corner world angle>) btheta2(<second corner view angle>)

    if the bbox projects to an empty screen range:
        return false

    x = <first bbox screen x>
    while x is inside the bbox screen range:
        bboxClipScan(x)
        if x is covered by a solid interval:
            x = <first x after the covered interval>
        else:
            return true
    return false
```

The `bbox.` family prefix is shortened to `b` (`bbox.x1` → `bx1`). The `bx*` /
`by*` corner markers each fold a `value` (range R0); the `bangle*` / `btheta*`
markers each fold an `angleValue`. Each `bx1/by1/bx2/by2` also carries the
`boxpos` region in its int slot so the corner pick is self-describing.

## Seg projection

Called once per seg between `R_AddLine` and `nextSeg`. A seg can be
skipped outright, project to no columns, or enter the visible-run scan.

```
procedure seg_projection(i):
    if the seg faces away or is an empty two-sided line:
        return                                  # only R_AddLine; on to the next seg

    angle1(<endpoint A world angle>) theta1(<endpoint A view angle>)
    angle2(<endpoint B world angle>) theta2(<endpoint B view angle>)

    if the projected seg covers no screen columns:
        nextSeg(i)                              # dead-end (empty range)
        return

    horizontal_visible_run_scan(i)              # may itself end with nextSeg(i)
```

`angle1`/`angle2` are world angles to the seg's two endpoints;
`theta1`/`theta2` are the same angles relative to the view. All four fold
an `angleValue`. The walls-are-vertical invariant holds: distance is
constant down a column, so per-column scale is a function of `theta` and
the column, computed once at `setCursorX`.

## Horizontal visible-run scan

Walks the projected screen range left to right, finding each maximal
uncovered run. Each run becomes one stored wall range.

```
procedure horizontal_visible_run_scan(i):
    x = <first projected x>
    while x is inside the projected seg range:
        clipScan(x)
        if x is covered by a solid interval:
            x = <first x after the covered interval>
            if x is now past the range end:
                clipScan(x)                      # final closing scan -> seg dead-ends (nextSeg)
                return
            continue
        ds.x2(<run stop>)
        wall_range_setup(i, x, <run stop>)
        x = <first x after the stored range>     # if past the end, the seg closes without nextSeg
```

`clipScan(x)` is the semantic source of the visible-run start `x1`;
`ds.x2` (the `drawseg.x2` marker, prefix shortened to `ds.`) carries the run's
right edge `x2`. `R_StoreWallRange` (below)
repeats neither — consumers recover them from these rows. Solid and
closed-door segs extend the solid-interval set; open portals clip against
it but do not extend it.

## Wall-range setup

Called once per uncovered run. Emits the drawseg record, the per-part
texture-mid values, the scale/silhouette payloads, the u-phase angle,
the per-plane visibility checks, then the column loop.

```
procedure wall_range_setup(i, x1, x2):
    R_StoreWallRange(i)
    segKpart(<part mask>)                        # slot `pat`; decoded: none | mid | mid+upper | ... (4*mid+2*upper+lower)

    segDcTmidMid(<middle dc_texturemid>)         # value, R3
    segDcTmidUpper(<upper dc_texturemid>)        # value, R4
    segDcTmidLower(<lower dc_texturemid>)        # value, R4

    ds.meta(i=i, wall_kind=<solid|closed|portal>, silhouette=<none|bottom|top|both>)

    ds.scale1.den(<den>)   ds.scale1(<scale at x1>)    # `.den` pairs with its value
    ds.scale2.den(<den>)   ds.scale2(<scale at x2>)
    ds.scalestep.den(<den>) ds.scalestep(<per-column step>)
    ds.bsilheight(<bottom silhouette height>)
    ds.tsilheight(<top silhouette height>)

    ds.uPhase(<view_angle - rw_normalangle>)   # angleValue

    for kind in the plane(s) this range opens (ceiling, floor):
        for vp in 0 .. <selected instance>:
            R_CheckPlane(<kind>, vp)             # kind positional (real DOOM call; not folded)
        R_CheckPlane.result(<kind>, <selected instance>)

    wall_column_loop(i, x1, x2)
```

`segKpart`'s `pat` indexes the K-part tables that map a span ordinal
(0,1,2) to a wall part (mid / upper / lower) — the wall-column loop uses
it to know which part each `wallSpanMeta` ordinal refers to. The `drawseg.`
family prefix is shortened to `ds.` (`drawseg.scale1` → `ds.scale1`); the eight
`ds.scale*` / `ds.*silheight` markers each fold a `value` (ranges R5–R9);
`ds.uPhase` folds an `angleValue`.

`R_CheckPlane` is the runtime-visplane assignment: for each plane the range
opens, it scans visplane instances `vp = 0, 1, ...` until it reaches the
one this range belongs to, and `R_CheckPlane.result` records that
selection. It runs *after* `ds.uPhase` and *before* the first
column's `setCursorX`. (`R_CheckPlane` corresponds to DOOM's
`R_CheckPlane`/`R_FindPlane`.)

## Wall-column loop

Each screen column of the range emits the column's u-index and scale, a
staged ceiling clip, the floor/ceiling plane marks, the visible wall
spans, and the column's clip update.

```
procedure wall_column_loop(i, x1, x2):
    for x in x1 .. x2:
        setCursorX(x)
        wallColU(u_idx=<texture column>, <encoded column scale>)   # folds value, R5
        screenY(<staged ceiling clip>)

        # body: plane marks first (they read the prior clip), then wall spans
        for kind in [ceiling, floor]:
            if the plane mark is non-empty:
                <ceilMark|floorMark>(p=<plane>, vp=<instance>)     # planeMark, kind folded
                screenRange(y1=<mark y1>, y2=<mark y2>)

        for each visible span, in K-order:
            wallSpanMeta(y=<span y1>, ordinal=<K-order index>)
            setCursorY(<span y1>, <texel-top v0>)                  # folds value, R3
            pixel(<palette index>, <w>)   x span_height

        if the column's clip moved:
            clipUpdate
            screenRange(y1=<new ceiling clip>, y2=<new floor clip>)
```

`setCursorX`, `wallColU`, `setCursorY`, and `pixel` carry no seg id or
range endpoints — they are scoped by the active `R_StoreWallRange` /
`ds.x2` and the current column. `planeMark` omits `x`; the extracted
mark column is the active `setCursorX`.

`wallSpanMeta(y, ordinal)` replaces the old scheme of recovering span
identity by matching `setCursorY` against span starts: the ordinal names
which K-order span this is directly, and `segKpart`'s table maps it to the
mid/upper/lower part. `setCursorY` then folds the span's texel-top `v0`
value; each `pixel` carries the palette index as AR feedback and a paint
width `w` (1 = high-detail, 2 = low-detail). RGB decode stays host-side.

## Flat (visplane) pass

After the walls, fill floor/ceiling pixels for the visplane instances the
wall columns assigned via `R_CheckPlane`. This replaces a direct `done`
after traversal. The shared cursor flips once, up front, from the wall
pass's downward advance to the rightward advance flats use.

```
procedure flat_pass():                       # R_DrawPlanes
    R_DrawPlanes
    setCursorDirectionX                       # flat pixels advance rightward
    R_DrawPlanes.nextPlane(p=-1)              # prime the plane walk

    for each used physical plane p, in order:
        R_DrawPlanes.nextVp(p=p, vp=-1)       # prime this plane's instance walk
        for each used visplane instance vp of p:
            visplaneBegin(p=p, vp=vp)
            if p is a sky plane:
                R_DrawPlanes.nextVp(p=p, vp=vp)   # visited and skipped
                continue
            make_spans(p, vp)
            R_DrawPlanes.nextVp(p=p, vp=vp)       # close this instance
        R_DrawPlanes.nextPlane(p=p)               # close this plane
    # falls through to the weapon pass (or `done` if the HUD is off)
```

The walk is sentinel-driven: a `nextPlane` / `nextVp` past the last used
plane / instance advances the parent loop; `visplaneBegin` for a sky plane
emits the next `nextVp` immediately, skipping the fill.

```
procedure make_spans(p, vp):                  # R_MakeSpans
    for x from the visplane's first column to its last + 1:
        R_MakeSpans.col(x)
        for each span slot (top, bottom) that closes at x:
            R_MakeSpans.closeSlot(slot=<0|1>)
            for each scanline y in the closed run:
                map_plane(y, <run x1>, <run x2>)

procedure map_plane(y, x1, x2):               # R_MapPlane: one horizontal run at fixed y
    R_MapPlane.row(y)
    setCursorY(y)                             # no folded value here (next is a cursor)
    setCursorX(x1)                            # R_MapPlane setup from setCursorX's derived columns
    pixel(<lit flat palette index>, <w>)   x span_width
```

A visplane's per-column `(top, bottom)` extents become horizontal spans
(a maximal run of columns sharing a scanline). Walking columns left to
right, the runs that *close* at column `x` are derived from the per-column
tops/bottoms at `x-1` and `x`; each closing run is drawn as one row. Each
flat `pixel` advances the R_MapPlane setup by its offset, floor-mods to
`(u, v)` in the native 64×64 flat tile, looks up the palette index, and
applies the row's `zlight` colormap. Flat texels come from a compiled
lookup table; prefill carries only flat ids — no atlas or texel tokens.

Note that `setCursorY` here does *not* fold a value (its next token is
`setCursorX`, not a carrier) — the same marker that folds a texel-top in
the wall-column loop is a bare cursor move here.

## Weapon pass (player sprite)

When the HUD is enabled, the flat pass flows into the weapon phase instead
of `done`. The cursor flips back to downward, and the baked weapon sprite
is blitted column-major over the 3D scene (last-write-wins). Transparent
texels are skipped by advancing the cursor without emitting a pixel.

```
procedure weapon_pass():                      # R_DrawPlayerSprites
    R_DrawPlayerSprites
    setCursorDirectionY                       # back to downward for the sprite
    for col in the sprite's column range:
        setCursorX(<col>)
        setCursorY(<sprite top>)
        for row in the sprite's rows:
            if the texel is transparent:
                setCursorY(<row + 1>)          # skip: advance, no pixel
            else:
                pixel(<palette index>, <w>)
    setCursorX(<one past last column>)
    # falls through to the status-bar pass (or `done` if no draw list)
```

## Status-bar pass (HUD)

Last, the status bar is composited as a draw-list of `V_DrawPatch` calls —
one `ST_Drawer.item` per patch, in DOOM's painter order, each blitted 1:1
(raw, unscaled, unlit) so later widgets overwrite the plate beneath them.

```
procedure status_bar_pass():                  # ST_Drawer
    ST_Drawer
    for i in the draw list, in painter order:
        ST_Drawer.item(item=i)
        setCursorX(<patch origin x>)
        for col in the patch's columns:
            setCursorY(<patch origin y>)
            for v in the patch's rows:
                if the texel is transparent:
                    setCursorY(<origin y + v + 1>)
                else:
                    pixel(<palette index>, 1)
            if col is not the last:
                setCursorX(<origin x + col + 1>)
    done
```

`ST_Drawer.item(item=i)` indexes the baked draw-list tables (patch id +
screen origin/size); each patch is its own masked lump, composited one at
a time. The bar is *not* the weapon's path — it is the unscaled
`V_DrawPatch` blitter.

## Terminal

The top-level driver runs, in order:

```
procedure render():
    setCursorDirectionY        # AR entry
    side_bit_setup_pass()
    visit_node(<root node>, 0)  # no-BSP maps enter subsector 0 directly
    flat_pass()
    weapon_pass()              # only if the HUD is enabled
    status_bar_pass()          # only if a HUD draw list exists; emits `done`
```

`done` is the single terminal token. With the HUD disabled, the flat pass
emits `done` directly (and the weapon / status-bar passes are absent).
