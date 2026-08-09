# BSP traversal and wall order in `torchwright_doom`

This is an intermediate technical brief for the blog post, not proposed final
copy. It keeps the facts and distinctions that the eventual explanation should
preserve, while marking which implementation details deserve a real deep dive
and which ones will probably distract from the story.

The central fact is simple but easy to state incorrectly:

> `torchwright_doom` does not traverse the entire BSP, collect the walls, and
> then render them in the resulting order. It traverses the BSP front-to-back
> and renders walls as it reaches the leaf subsectors. Rendering the nearer
> subtree changes the occlusion state used to decide whether the farther
> subtree needs to be visited at all.

That coupling between traversal and rendering is the heart of the algorithm.
It is also the part most worth explaining in the blog.

## Corrections to the current draft

The current draft has the right broad ingredients—a flat map, sector heights,
a precomputed BSP, partition lines, and convex subsector leaves—but three
statements need to change before the prose is developed further.

First, the engine visits the **nearer** side of a BSP partition first, not the
farther side. A generic BSP can be traversed back-to-front for a painter's
algorithm, which is probably the source of the confusion. Doom instead walks
front-to-back and uses the screen coverage established by nearer walls to clip
or reject farther geometry.

Second, `torchwright_doom` does **not** finish the BSP traversal before it starts
rendering walls. It performs a short side-bit setup pass first, but the actual
tree traversal and wall rendering are interleaved. When traversal reaches a
subsector, that subsector's wall segs are projected, clipped, and rasterized
before traversal returns to the parent node and considers the other child.

Third, a subsector is not simply painted as one convex polygon. Its segs are
processed in stored order. Each facing, non-empty seg is projected into screen
columns; visible horizontal fragments become wall ranges; and those ranges are
drawn column by column. Floor and ceiling regions are recorded during this
wall pass and filled later by the visplane pass.

The opening description of Doom's map format could also be made slightly more
precise. “A Doom level contains no 3D geometry” is an effective slogan, but the
technically safer claim is that it contains no arbitrary 3D mesh. The map is a
2D arrangement of vertices, linedefs, sidedefs, and sectors, while sector floor
and ceiling heights supply the vertical extent. Walls are inferred where sector
boundaries are projected into the view. This distinction is worth one sentence,
not a detour into every WAD lump.

## The level of detail appropriate for the blog

The BSP section should probably explain four layers of the system:

1. Why a BSP gives a view-dependent front-to-back traversal without sorting
   every wall by distance.
2. Why Doom must draw and update occlusion while it traverses, rather than
   merely extracting a wall list from the tree.
3. How `torchwright_doom` serializes recursion into an autoregressive token
   protocol.
4. How attention turns the generated history into the tables, stack, and
   occlusion memory that ordinary code would keep in mutable data structures.

That is enough detail to make the transformer implementation feel real rather
than magical. The exact wall-projection arithmetic, texture coordinates,
lighting, visplane merging, and numerical softmax margins belong elsewhere.

## What the BSP contributes

The BSP is precomputed with the map and represented as a binary tree. Each
internal node contains:

- a 2D partition line, stored as an origin `(px, py)` and direction `(dx, dy)`;
- two child references, each referring to another node or a subsector leaf;
- a bounding box for each child.

The leaves are subsectors. Doom's node builder has partitioned the map until
each leaf region is convex, and the subsector points to a contiguous run of
wall segs. The BSP therefore orders regions of space. It does not contain a
pre-sorted list of wall pixels, and the runtime never compares every pair of
walls by Euclidean distance.

In `torchwright_doom`, the BSP nodes, child references, bounding boxes,
subsectors, and segs are serialized into the scene prompt. They become
attention-addressable scene facts in the compiled model. The root is the last
node in the prompt, following Doom's map convention.

The production prompt contains a fixed, cropped region of the map rather than
all of E1M1. That preprocessing retains the minimal BSP subtree reaching the
selected subsectors and redirects pruned branches to an empty subsector. This
is useful background for a broader architecture discussion, but it is not part
of the per-view wall-order algorithm and probably should not appear in the main
BSP explanation.

## Phase one: precompute the viewer side of every node

Before walking the tree, `torchwright_doom` runs a small setup protocol over
the BSP nodes. For each node it computes which side of the partition contains
the viewer:

```text
side_raw = dy * (viewx - px) - dx * (viewy - py)
side = sign(side_raw)
```

The graph emits two tokens per node: an `R_PointOnSide(node)` request followed
by an internal `pointOnSideResult(node, side)` record. In raw and pretty text,
that result appears as either `frontSideResult(node)` or
`backSideResult(node)`. Those result rows form a lookup table keyed by node ID.
Traversal can later ask for `side(node)` through an attention read.

This prepass is specific to the autoregressive implementation. Ordinary Doom
can call `R_PointOnSide` while recursively visiting a node. Here, emitting and
re-embedding the result turns the side bit into a shallow, stable piece of
history that any later token can retrieve. It also keeps the tree-walk steps
small enough to compile cleanly.

The main blog should mention that there is a side-bit setup pass, because the
current draft otherwise implies that each tree node performs all of its
geometry inline. The cross-product formula itself is optional: it is a good
small technical detail if the article wants to show that the “which side?”
test is ordinary geometry compiled into the model, but the derivation is not
necessary.

If a map has no BSP nodes, the engine skips this pass and directly enters
subsector zero. That edge case can be omitted from normal prose.

## Phase two: walk the near child first

At an internal node, the stored child names are static, but their runtime roles
are view-dependent. The precomputed side bit selects one child as the first,
viewer-side child and the other as the second, opposite-side child:

```text
if side(node):
    first  = node.front_child
    second = node.back_child
else:
    first  = node.back_child
    second = node.front_child
```

The first child is entered unconditionally. Recursively applying this choice
produces a front-to-back traversal of the BSP regions. “Front-to-back” here is
a property of the partition tree, not a promise that every individual seg is
globally sorted by its scalar distance from the camera. Within a subsector,
segs remain in their stored order.

The logical traversal is:

```text
visit_node(node, depth):
    visit(first child, depth + 1)

    if second child's bounding box may still be visible:
        visit(second child, depth + 1)

    return to parent
```

The important timing is hidden by this compact pseudocode: visiting a child
includes completely processing the visible wall ranges found below it. By the
time the first child returns, nearer opaque wall fragments have already updated
the horizontal coverage state, and portal wall columns have updated their
vertical clips.

## What happens in a subsector

Reaching a subsector switches from tree traversal to its seg loop. The engine
processes the subsector's contiguous seg range in stored order. For each seg it:

1. Rejects a back-facing seg or a statically empty two-sided line.
2. Computes view-relative angles to both endpoints.
3. Projects the endpoints into a screen-column range and rejects an empty or
   out-of-view range.
4. Clips that range horizontally against opaque wall fragments already drawn.
5. Turns each remaining uncovered run into an `R_StoreWallRange` operation.
6. Sets up and draws that range immediately, column by column.

One projected seg can produce several wall ranges. For example, if a nearer
opaque wall covers columns 30–40, a farther seg projected across columns 20–50
can produce ranges 20–29 and 41–50. It is not drawn through the covered middle.

The complete order below the BSP is therefore:

```text
BSP child order
    subsector order
        stored seg order
            visible fragments from left to right
                columns from left to right
                    wall parts (middle, upper, lower; absent parts skipped)
                        pixels from top to bottom
```

The main BSP section does not need the wall-part and pixel sub-order unless it
is making a broader point about the autoregressive output stream. The seg and
visible-fragment levels do matter, because they prevent “the BSP returns a
sorted list of walls” from becoming an accidental simplification.

## Horizontal occlusion: the reason near-first traversal works

Doom's `solidsegs` structure records horizontal screen intervals that have
been closed by opaque walls. `torchwright_doom` represents the same logical
state as a collection of prior `R_StoreWallRange` rows.

A one-sided wall or a closed two-sided door publishes its visible `[x1, x2]`
fragment as solid. An open portal clips against existing solid coverage but
does not become a solid horizontal interval itself, because geometry can still
be visible through the opening.

When scanning a later projected seg, the renderer repeatedly asks two
questions:

```text
Does a solid interval cover column x?
If not, where does the next solid interval begin?
```

If `x` is covered, the scan jumps to the end of the covering interval plus one.
If it is uncovered, the next visible run ends immediately before the nearest
later solid start, or at the seg's projected endpoint if no solid begins first.

This is more informative than simply saying that nearer walls are “drawn
first.” Their screen coverage becomes state that prevents farther walls from
emitting overlapping pixels. The final screen buffer is last-write-wins, so
front-to-back emission would be wrong without this suppression: farther pixels
would otherwise overwrite nearer ones.

The graph stores solid fragments without merging them into a mutable interval
list. Repeated attention queries over the union reproduce the behavior that
Doom gets from merged `solidsegs`. That implementation detail is distinctive
and may deserve a sidebar, but the main text only needs the logical union of
opaque screen intervals.

## Vertical clipping through portals

Horizontal solid coverage is enough for opaque walls, but not for portals. A
portal may have an upper wall tier, a lower wall tier, and an opening through
which farther geometry remains visible.

For each screen column, the renderer therefore also remembers a ceiling clip
and floor clip. An untouched column is vertically open. A solid wall closes
the column completely; an upper portal tier moves the ceiling clip down; and a
lower portal tier moves the floor clip up. A later wall at the same column is
clamped to the remaining open vertical range before it emits pixels.

This is adjacent to the BSP story rather than part of the tree traversal
itself, but it is worth at least one paragraph. Without it, “portals do not add
to `solidsegs`” leaves an obvious unanswered question about what stops farther
walls from overwriting the portal frame. The exact projected y-bound formulas,
clip sentinels, and upper/lower texture rules can be left to a wall-rasterization
section.

## Checking the second child's bounding box

After the first child has returned, the engine does not automatically traverse
the second child. It performs the equivalent of Doom's `R_CheckBBox` using the
second child's stored bounding box and the horizontal solid coverage produced
so far.

The check is:

1. Classify the viewer into one of the regions around the box. If the viewer is
   inside the box, conservatively visit the child.
2. Select the two extreme box corners from Doom's `checkcoord` table.
3. Project those corners to obtain the box's horizontal screen extent.
4. Scan that extent against the union of prior solid intervals.
5. Visit the child as soon as any uncovered column is found; prune it only if
   the entire projected extent is covered.

This test is deliberately conservative. It uses opaque horizontal coverage,
not the finer vertical clip state. It may allow traversal of a subtree whose
walls are later rejected by column clipping, but it must not discard geometry
that could be visible.

This is a strong candidate for a deeper paragraph in the final article. It
shows that the BSP is not just an ordering device: its child bounding boxes,
combined with occlusion accumulated from the near side, allow the engine to
skip whole regions of the map.

## How recursion becomes tokens

There is no Python call stack during inference. The traversal is serialized
into tokens whose surface names reflect the logical states. Schematically:

```text
bspFront(node=n, depth=d)                    # enter node; descend first child
bspCheckBack(node=n, depth=d)                # first child returned; test second
bspReturn(entity_u=node<n>|ss<s>, depth=d)   # node or subsector is finished
R_Subsector(s=s, depth=d)                    # enter a leaf
R_AddLine(i)                                 # process one seg in that leaf
```

A representative control sequence is:

```text
bspFront(root, 0)
    bspFront(first child, 1)
        R_Subsector(...)
        R_AddLine(...)
        ...wall ranges and pixels...
        bspReturn(subsector, 2)
    bspCheckBack(first child, 1)
    ...
    bspReturn(first child, 1)
bspCheckBack(root, 0)
...
bspReturn(root, 0)
R_DrawPlanes
```

The current token acts as the program counter. The compiled forward pass reads
that token, recovers whatever state the corresponding transition needs, and
emits the next token. The host merely feeds the emitted token back; it does not
perform the recursion or choose a child.

This token-level view is essential to a `torchwright_doom` article. The final
blog does not need to enumerate every projection or wall-column token, but it
should show at least the three traversal tokens and one small example of the
sequence.

## How the return stack uses attention

When a child emits `bspReturn`, the model must recover which parent invoked it
and whether it was that parent's first or second child. A static child-to-parent
table is not enough: the required answer is the active runtime frame, especially
when an empty subsector can be referenced from multiple pruned branches.

At every `bspFront` or `bspCheckBack` row, the model publishes a traversal-edge
record keyed by:

```text
(child entity ID, child tree depth)
```

The value stores:

```text
(parent node ID, child-was-first)
```

On `bspReturn(entity_u, depth)`, attention selects the most recent matching
edge. If the child was first, the next token is
`bspCheckBack(parent, depth-1)`. If it was second, the parent is also complete
and the model emits another `bspReturn`. A return at depth zero ends the BSP
walk and starts the plane pass.

This is probably the best attention deep dive for the BSP section. It is easy
to motivate, directly replaces a familiar data structure, and demonstrates
that attention is doing concrete random-access memory rather than vaguely
“reasoning about” the level.

## The other attention reads worth preserving

The traversal relies on several different memory patterns:

- **Static scene lookup.** Node and seg fields are recovered from prompt rows by
  exact ID matching. IDs are encoded with compact lifted keys rather than wide
  one-hots.
- **Side table.** A node ID retrieves the side bit emitted during the setup
  pass.
- **Most-recent context.** Marker attention recovers the active subsector, seg,
  wall range, bbox state, or column state from earlier protocol rows.
- **Traversal edge lookup.** The composite `(entity, depth)` key implements the
  dynamic return stack.
- **Solid interval queries.** Attention finds an interval covering a screen
  column and performs a bucketed successor search for the nearest later start.
- **Vertical clip lookup.** A column key retrieves the most recent ceiling and
  floor clip update for that column.

The final blog should not explain all six with equal weight. Static lookups and
the return stack establish the mental model. Solid interval attention is the
most interesting optional second example. The many marker/context reads can be
summarized as the ordinary bookkeeping that keeps a long autoregressive
protocol coherent.

It may be useful to say explicitly that this model is compiled, not trained to
discover BSP traversal. The query/key/value projections implement chosen
algorithms. Attention is being used as causal, content-addressable memory over
the prompt and generated history.

## Optional deep dive: interval membership in a dot product

If the article wants one compact piece of attention algebra, the horizontal
coverage test is a good candidate.

For a solid fragment `[x1, x2]`, the graph defines padded endpoints
`a = x1 - 1` and `b = x2 + 1`, then publishes the key:

```text
[-2, 2(a+b), -2ab]
```

A screen-column query `x` is represented as:

```text
[x^2, x, 1]
```

The dot product is:

```text
-2(x-a)(x-b)
```

It is positive inside the padded interval and loses to a constant sentinel
outside. The attention head therefore returns the coverage flag and the
covering interval's end. A separate radix/bucket attention search finds the
nearest interval start strictly after `x`.

This is excellent evidence that the transformer really contains the renderer,
but it is more detail than the main BSP narrative requires. It should be a
sidebar, an equation callout, or material saved for a deeper “attention as
data structures” section.

## Details to mention briefly

These facts matter for accuracy but should normally take only a clause or
sentence:

- The root is the last BSP node in the scene prompt.
- Node-or-subsector child references share one unified ID space.
- Segs within a subsector are processed in stored order.
- Back-facing and empty segs are discarded before wall projection.
- Open portals do not become horizontally solid; closed doors do.
- Bounding-box pruning happens only after the near child has affected
  occlusion state.
- Once the root returns, the renderer begins the floor/ceiling plane pass.
- The host applies emitted cursor and pixel tokens but makes no ordering or
  visibility decisions.

## Details to skip from the BSP section

The following are real and important elsewhere, but would make this particular
explanation harder to follow:

- The exact atan/octant and field-of-view projection implementation.
- Perspective scale denominators and the drawseg scalar sidecar sequence.
- Texture pegging, `dc_texturemid`, u/v coordinates, texture-bank lookup, and
  lighting.
- K-part lookup tables beyond saying that visible upper/middle/lower wall parts
  are emitted in a fixed order.
- Runtime visplane instance conflict detection and floor/ceiling span creation.
- Weapon and status-bar compositing.
- The prompt subset's mean-centering and ID-remapping mechanics.
- Vocabulary capacities such as the maximum number of nodes or maximum tree
  depth.
- Exact attention gains, RoPE dimensions, fp32 tolerances, and why production
  requires eager attention.
- Compiler scheduling, residual widths, layer counts, and attention-head
  packing.

Those numerical attention details are useful if the article later asks how a
softmax lookup remains reliable across a 50,000-token frame. They are not
needed to explain how wall order is chosen.

## Phrases to avoid in downstream prose

Avoid saying:

- “The far side is drawn first.” That describes a painter-style BSP traversal,
  not Doom's front-to-back traversal.
- “The BSP traversal produces a sorted wall list.” It produces a recursive
  region order, and wall processing is interleaved with that traversal.
- “`torchwright_doom` traverses the BSP first and renders afterward.” Only the
  side-bit setup happens before traversal; walls are rendered at the leaves.
- “The BSP sorts every wall by distance.” It orders partitioned regions. Seg
  order and screen clipping finish the job.
- “A subsector is painted directly.” Its segs are culled, projected, clipped,
  split into ranges, and rasterized.
- “Attention learns the wall order.” The renderer graph explicitly compiles
  the side tests, transition rules, and memory queries into transformer
  weights.
- “Near walls overwrite far walls.” Walls are emitted near-to-far; far overlap
  is suppressed before emission. Later weapon/HUD compositing is a separate
  last-write-wins use.

## A compact factual spine for later drafting

The eventual prose can be built around this sequence:

1. Doom's map is horizontally two-dimensional, with sector heights supplying
   the vertical structure.
2. Its precomputed BSP recursively divides the map into convex subsectors.
3. `torchwright_doom` puts that BSP and the wall segs into the scene prompt.
4. The model first emits one viewer-side result for every BSP node.
5. It enters the root and recursively visits the viewer-side child first.
6. On reaching a subsector, it immediately projects and draws visible wall
   fragments.
7. Opaque fragments become horizontal coverage; portal tiers update vertical
   per-column clips.
8. After the near child returns, the model projects the far child's bounding
   box and skips that whole subtree if prior opaque coverage hides it.
9. `bspFront`, `bspCheckBack`, and `bspReturn` serialize recursion into the
   generated token stream.
10. Attention retrieves scene records, side bits, runtime parent frames, solid
    intervals, and the latest clip state from the causal history.

If those ten facts survive downstream editing, the resulting blog explanation
will be both compact and faithful.

## Source map

The implementation paths most relevant to this writeup are:

- `torchwright_doom/model/traversal/bsp_traversal.py` — side prepass, child
  selection, node/subsector dispatch, and top-level traversal transitions.
- `torchwright_doom/model/traversal/traversal_edges.py` — attention-backed
  runtime return frames.
- `torchwright_doom/model/traversal/bbox_pruning.py` — second-child
  `R_CheckBBox` protocol.
- `torchwright_doom/model/traversal/solid_intervals.py` — horizontal opaque
  coverage and successor queries.
- `torchwright_doom/model/raster/seg_scanner.py` — subsector seg order,
  backface culling, projection protocol, and visible-run scan.
- `torchwright_doom/model/raster/wall_column_state.py` — per-column vertical
  clip memory and wall-span clipping.
- `torchwright_doom/model/raster/wall_column_renderer.py` — column/span/pixel
  control order.
- `torchwright_doom/model/past.py` and
  `torchwright_doom/model/attention_handles.py` — attention memory primitives.
- `torchwright_doom/pydoom/renderer.py` and
  `torchwright_doom/pydoom/drafter.py` — the plain-Python rendering and token
  protocol oracles.
- `PROTOCOL.md` — the readable end-to-end token protocol.
