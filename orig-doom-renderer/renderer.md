# DOOM Software 3D Renderer

A reference describing how the original DOOM (1993) software 3D renderer works. All key invariants, data structures, and algorithms are captured here so that the renderer can be understood end-to-end without external assets.

## What it renders

Three categories of geometry, in this fixed order:

1. **Segments** — walls and portals. Always strictly vertical.
2. **Flats** — ceilings and floors. Always strictly horizontal.
3. **Masked** ("things") — monsters, items, the player's weapon, and partially-transparent walls.

The defining property: **walls and flats are rendered with zero overdraw** (each pixel is written exactly once). Sprites and transparent walls introduce minimal overdraw.

## Top-level entry point

```c
void R_RenderPlayerView (player_t *player) {
  R_RenderBSPNode (numnodes-1); // root node is last
  R_DrawPlanes ();              // Draw visplanes
  R_DrawMasked ();
}
```

The frame proceeds as:

1. **`R_RenderBSPNode`** — traverses the BSP front-to-back. As it visits subsector leaves it draws walls/portals into the framebuffer, populates `visplanes[]` for ceiling/floor regions, and queues `vissprites[]` for things in each subsector.
2. **`R_DrawPlanes`** — fills in the ceilings/floors recorded in visplanes.
3. **`R_DrawMasked`** — sorts and draws sprites and transparent walls back-to-front, then draws the player's weapon sprite on top.

The framebuffer is **never cleared**; geometry coverage is exhaustive (BSP-driven walls + flats fully tile the visible area before masked rendering begins).

## Key invariants

These facts are load-bearing — most of the renderer's tricks rely on them.

- **Walls are vertical** ⇒ distance from the player is constant along a screen-column ⇒ perspective-correct texture mapping needs only one expensive computation per column; pixels within a column use cheap linear interpolation.
- **Floors and ceilings are horizontal** ⇒ distance is constant along a screen horizontal span ⇒ flats are rendered as horizontal spans (not columns) for the same reason.
- **Subsectors are convex** ⇒ segments inside a subsector can be drawn in any order; only inter-subsector ordering matters, and the BSP provides that.
- **The BSP is precomputed offline** ⇒ wall positions are baked at build time and cannot move at runtime.
- **The framebuffer is never cleared** between frames — relies on full coverage.
- **Walls, portals, and flats are zero-overdraw**; only masked elements may overdraw.

## Binary Space Partitioning (BSP)

### Why BSP

Earlier versions of the engine operated on raw map sectors. Starting in the player's sector, it would look for double-sided lines (portals) and recurse into adjacent sectors. This worked for convex sectors and simple cases, but broke down on:

- **Concave sectors** — bookkeeping for what had been visited became expensive.
- **Nested sectors** — a sector containing another sector required complex stack management.
- **Pathological geometry** — John Romero's circular-stairs in E1M2 made the sector-list builder explode in runtime.

Carmack adapted Bruce Naylor's 1993 paper *"Constructing Good Partitioning Trees"* and first applied BSP to the SNES port of Wolfenstein 3D (axis-aligned everything, gentler test bed) before bringing it to DOOM.

The BSP transforms the polygonal map into a binary tree such that **traversing it yields a front-to-back ordering of all subsectors at the cost of a constant number of side tests** (one per tree level), regardless of camera position.

### How a BSP is built

Recursively pick a splitter line and split the map in two:

- Lines crossed by the splitter are **split into segments (`SEGS`)**.
- Sectors crossed by the splitter become **sub-sectors (`SSECTORS`)**.
- A subsector is a **leaf** once it is convex; otherwise recurse.

Splitter selection matters. A good splitter divides the map evenly (limits tree depth) and is axis-aligned (cheaper side tests, easier debugging). DOOM's `doombsp` tool scores every line in a subspace and picks the highest-scorer. This is CPU-intensive: ~8 seconds for E1M1, ~11 minutes for all 30 maps of `DOOM.WAD` on a NeXTstation Turbo.

### Worked example

Consider a single sector forming a square room with an internal pillar (a "donut" — one sector with a hole). It has 8 vertices and 8 single-sided lines (each line has one side, on its right):

- Outer walls: `A`, `B`, `C`, `D`
- Pillar walls: `E`, `F`, `G`, `H`

**Step 1: split on H.**
- Lines crossing H are split: `B` → `B1`,`B2`; `D` → `D1`,`D2`.
- Resulting subsectors:
  - `{A, B1, H, D1}` — **convex**, becomes a leaf.
  - `{E, F, G, D2, C, B2}` — **concave**, recurse.

**Step 2: split on G** (within the concave subsector). Some lines split further; one side becomes convex.

**Step 3: split on F** (within what remains concave).

After three splits, all leaves are convex. The BSP has H at the root, G and F below, and four convex subsectors at the leaves. Total geometry has grown ~50% (from line splits) but any camera position can be sorted with three side tests.

### Traversal

```c
void R_RenderBSPNode (int bspnum)
{
    node_t*    bsp;
    int        side;

    // Found a subsector?
    if (bspnum & NF_SUBSECTOR)
    {
      if (bspnum == -1)
        R_Subsector (0);
      else
        R_Subsector (bspnum&(~NF_SUBSECTOR));
      return;
    }

    bsp = &nodes[bspnum];

    // Decide which side the view point is on.
    side = R_PointOnSide (viewx, viewy, bsp);

    // Recursively divide front space.
    R_RenderBSPNode (bsp->children[side]);

    // Possibly divide back space.
    if (R_CheckBBox (bsp->bbox[side^1]))
        R_RenderBSPNode (bsp->children[side^1]);
}
```

Walking the example tree from a viewpoint **P1** that lies right-of-H, left-of-G, left-of-F yields the front-to-back subsector order `1,2,3,4`. From a viewpoint **P2** that lies left-of-H, left-of-G, right-of-F yields `3,2,4,1`. Order *within* a subsector is irrelevant because subsectors are convex.

Note the asymmetry in the traversal code: the front child is always recursed into; the back child is recursed only if `R_CheckBBox` says its bounding box might still be visible. `R_CheckBBox` tests the bbox two ways:

1. Against the view frustum (skip if off-screen).
2. Against the current `solidsegs` occlusion array (skip if the bbox's screen-space X range is already fully covered by walls drawn so far).

The second test is what makes BSP rendering achieve cost proportional to the *visible* scene rather than the entire map: as front-space walls fill in `solidsegs`, whole back-space subtrees get pruned cheaply.

### Node format

Splitting lines are stored as `(point, delta)` rather than `(point1, point2)` — this makes the cross-product in `R_PointOnSide` faster because one of the vectors is already cached:

```c
typedef struct {
  fixed_t    x,y,dx,dy;       // partition line
  fixed_t    bbox[2][4];      // child bounding box
  unsigned short children[2]; // NF_SUBSECTOR = subsector
} node_t;
```

### Side test

Conceptually: the plane equation `ax + by + d = 0` evaluated at point `P` gives a signed distance — its sign tells you which side. DOOM avoids that arithmetic via a **2D cross-product**:

```c
int R_PointOnSide(fixed_t x, fixed_t y, node_t* node){
    fixed_t    dx, dy, left, right;

    if (!node->dx) { // Shortcut if node is vertical.
      if (x <= node->x) {
        return node->dy > 0;
      }
      return node->dy < 0;
    }

    if (!node->dy) { // Shortcut if node is horizontal.
      if (y <= node->y) {
        return node->dx < 0;
      }
      return node->dx > 0;
    }
    // Calculate node to POV vector
    dx = (x - node->x);
    dy = (y - node->y);

    if ( (node->dy ^ node->dx ^ dx ^ dy)&0x80000000 ) {
      if  ( (node->dy ^ dx) & 0x80000000 ) {
        // (left is negative)
        return 1;
      }
      return 0;
    }
    // Cross product here
    left = FixedMul ( node->dy>>FRACBITS , dx );
    right = FixedMul ( dy , node->dx>>FRACBITS );

    if (right < left) { // front side
      return 0;
    }
    return 1; // back side
}
```

Axis-aligned splitters get cheap shortcuts; the general path computes `(node.dy * dx) - (node.dx * dy)` and returns the sign. The XOR trick is a sign-only fast path that avoids the multiply when the result's sign is determinable from operand signs alone.

### Side effects of using a BSP

- **Preprocessing time and latency** for map designers (`doombsp` runs on every save).
- **Walls are baked**: BSP creates new vertices and segments. There is no general way to move walls at runtime — a hard limit on level dynamism.

## Wall rendering

After `R_RenderBSPNode` reaches a leaf, `R_Subsector` iterates over the segments in that subsector. The pipeline for each segment is: **project → angle-cull → horizontal-clip → vertical-clip → emit columns**.

### Projection: BAM angles

DOOM uses **BAM (Binary Angular Measurement)** — angles `[0°, 360°)` mapped to the full range of a 32-bit unsigned integer:

```c
// Binary Angle Measument, BAM.
#define ANG45   0x20000000
#define ANG90   0x40000000
#define ANG180  0x80000000
#define ANG270  0xc0000000
typedef unsigned angle_t;
```

For each segment endpoint:

1. Compute its angle relative to the player using `arctan(O/A)`.
2. **Backface cull**: if `angle1 - angle2 < 0`, the segment faces away — skip it.
3. Right-shift the 32-bit BAM down to 13 bits (4096-entry index).
4. Look up screen-space `X` in `viewangletox[4096]` (a precomputed startup table).

`viewangletox` is built so the player gets a 90° field of view; angles outside ±45° clamp to the screen edges.

The engine now has both endpoints' screen-space `X` and the segment's distance `z`. Clipping happens before drawing.

### Two segment types

- **Walls** — single-sided segments connecting one sector. Opaque. Have only a *middle texture*.
- **Portals** — two-sided segments connecting two sectors. The middle texture is usually absent (so you can see through), but they have *upper* and *lower* textures used when the adjacent sectors have different ceiling/floor heights (steps, windows).

### Horizontal clipping (crude pass)

Tracks horizontal occlusion in a `solidsegs` array. Only walls write to it; portals don't (they're see-through). Segments come out as **fragments** because clipping can split them.

```c
typedef struct {
    int first;
    int last;
} cliprange_t;

cliprange_t*    newend;
cliprange_t     solidsegs[32];
```

Each `cliprange_t` is one contiguous run of solid (occluded) screen columns. Initially the array represents only the off-screen sentinels; rendered walls add or merge entries.

#### Step-by-step example

A single-sector room with four walls `A`, `B`, `C`, `D`, player facing north.

**Initial state** — only off-screen sentinels:

```text
solidsegs[0] first = -0x7fffffff
             last  =          -1
solidsegs[1] first =         320
             last  =  0x7fffffff
```

**After A** — fits mid-screen, nothing occludes it; new entry inserted between sentinels:

```text
solidsegs[0] first = -0x7fffffff
             last  =          -1
solidsegs[1] first =         100
             last  =         220
solidsegs[2] first =         320
             last  =  0x7fffffff
```

**After B** — touches the left edge; left side is clamped via angle adjustment. No new entry needed; sentinel `solidsegs[0]` extends rightward:

```text
solidsegs[0] first = -0x7fffffff
             last  =          50
solidsegs[1] first =         100
             last  =         220
solidsegs[2] first =         320
             last  =  0x7fffffff
```

**After C** — sits just right of B, fully visible, extends `solidsegs[0]`:

```text
solidsegs[0] first = -0x7fffffff
             last  =          70
solidsegs[1] first =         100
             last  =         220
solidsegs[2] first =         320
             last  =  0x7fffffff
```

**After D** — bridges across, forces a merge of all entries into the universal sentinel:

```text
solidsegs[0] first = -0x7fffffff
             last  =  0x7fffffff
```

A single entry from `-INF` to `+INF` means the screen is fully horizontally occluded — the renderer can early-out cheaply on subsequent segments. The data structure uses minimal RAM (32 ranges × 8 bytes = 256 bytes) and is fast to query.

### Vertical clipping (fine pass)

Tracks per-column vertical occlusion using two arrays:

```c
#define SCREENWIDTH  320
#define SCREENHEIGHT 200
// clip values are the solid pixel bounding the range
// floorclip starts out SCREENHEIGHT
// ceilingclip starts out -1
short floorclip[SCREENWIDTH];
short ceilingclip[SCREENWIDTH];
```

For each column:

- `ceilingclip[x]` = highest solid pixel from the top (grows downward as ceilings extend).
- `floorclip[x]` = lowest solid pixel from the bottom (shrinks upward as floors extend).
- A column is **fully opaque** when `ceilingclip[x] == floorclip[x]`.
- **Walls** mark every column they cover as fully opaque.
- **Portals** only update the columns where their upper and/or lower textures actually paint pixels — leaving a vertical gap of see-through space in between.

#### Step-by-step example: 3-sector room

Three sectors `1`, `2`, `3` arranged near-to-far. Sectors 1 and 3 share the same ceiling/floor heights; sector 2 has a higher floor and lower ceiling, so it looks like a window between rooms 1 and 3.

Subsectors are visited in order `1`, `2`, `3`. Their segments:

- Subsector 1: walls `A`, `C`, `D`; portal `B` (1↔2).
- Subsector 2: wall `E`; portal `F` (2↔3).
- Subsector 3: walls `G`, `H`.

What each segment does to the occlusion arrays:

- **A, C, D** (walls in subsector 1): mark every column they cover as fully opaque.
- **B** (portal, no middle texture): renders an *upper* texture (because sector 2's ceiling is lower) and a *lower* texture (because sector 2's floor is higher). The occlusion arrays are advanced from the top by the upper texture's bottom edge and from the bottom by the lower texture's top edge — leaving a window-shaped see-through gap.
- **E** (wall in subsector 2): sits inside the window opening; marks its columns fully opaque.
- **F** (portal from sector 2 to sector 3, where sector 3 has *higher* ceiling and *lower* floor than sector 2): no upper or lower texture is rendered. The general rule is that upper/lower textures only render when the adjacent sector *closes in* (lower ceiling = needs upper texture, higher floor = needs lower texture); when the adjacent sector *opens up*, no texture is needed because there's nothing to bridge. The occlusion arrays are still updated to record what was "drawn" through F's middle. *This is also where visplane generation cares — see next section.*
- **G, H** (walls in subsector 3): seen through F's opening; finish marking the screen opaque. The crude horizontal pass had already clipped them to the visible window space.

### Emitting columns

After all clipping, walls and portals are rendered. For each fragment:

- A screen-space `Y` offset is computed from sector floor height and player distance.
- A column height is computed from ceiling/floor heights and distance.
- These values are linearly interpolated across the fragment's `X` range to generate a column for each screen-space `X`.
- Portals emit up to three texture types per column (upper / lower / middle); walls emit only middle.
- The column itself is drawn via the `colfunc` function pointer (one of several variants — solid, masked, fuzz for spectres, translated for player colors).

### Subpixel accuracy

Most contemporary engines snapped triangle vertex coordinates to integer pixels (`floor(A)`, `floor(B)`) before interpolating. DOOM **keeps the fractional part** while interpolating between vertices.

Example: walking from `A=(0.7, 0.7)` to `B=(5.3, 3.6)`:

- *Pixel-accurate*: discard fractions, walk `(0,0)` → `(5,3)`.
- *Subpixel-accurate*: navigate from the true `A` to the true `B`, only floor()ing when selecting a pixel to write.

Static, the difference looks small. **In motion**, when `A` shifts by `0.3` units, the pixel-accurate method may flip 5 pixels; the subpixel-accurate method flips 1. Surfaces feel solid instead of jiggling.

> Almost every other texture mapped game back then snapped triangle vertexes to integral pixel values, which meant that the individual texels in a surface would constantly be jumping around by up to a pixel from even tiny movements. Basically everything only feels loosely connected, and wiggles around a bit. DOOM did not have that problem.
> — John Carmack

### Perspective-correct texture mapping

Two formulas in play:

**Affine** (linear in screen space, fast, distorts at angles):

$$u_{\alpha} = (1-\alpha)\,u_{0} + \alpha\, u_{1} \quad \text{where } 0 \le \alpha \le 1$$

**Perspective-correct** (uses `1/z`, slow but right):

$$u_{\alpha} = \dfrac{(1-\alpha)\dfrac{u_{0}}{z_{0}} + \alpha\dfrac{u_{1}}{z_{1}}}{(1-\alpha)\dfrac{1}{z_{0}} + \alpha\dfrac{1}{z_{1}}} \quad \text{where } 0 \le \alpha \le 1$$

DOOM's invariant: **walls are always vertical**. This means `z` is constant *along a screen column*. So:

- The expensive perspective-correct math runs **once per column** (to get the texture coordinates at the column).
- Within a column, the renderer uses cheap linear interpolation along the texture's `v` axis.
- Net cost equals affine; visual result is perspective-correct.

The same trick applies to flats: along a horizontal screen span, `z` is constant on a horizontal plane, so flats are drawn as horizontal spans with one perspective-correct computation per span.

A general quad would have required up to six perspective-correct computations per pixel — far too expensive. By forbidding sloped walls, DOOM dodged this entirely.

Carmack vetoed any port that fell back to affine texturing. Era hardware-accelerated consoles (PlayStation, 3DO, Saturn) lacked perspective correction, so their ports either subdivided triangles or avoided texturing (Crash Bandicoot used Gouraud shading for this reason). The N64 had perspective correction thanks to SGI.

## Flats (visplanes)

After wall rendering, the framebuffer has walls drawn but ceilings and floors are gaps. The renderer fills them using **visplanes**, which were *recorded as a side effect of wall rendering*.

### Concept

A visplane describes a screen-space region representing **either a single ceiling or a single floor**. It has:

- A **height** (world-space).
- A **picnum** (texture).
- A **light level**.
- Two arrays as wide as the screen, `top[SCREENWIDTH]` and `bottom[SCREENWIDTH]`, defining one vertical column of coverage per `X`.

```c
// Now what is a visplane, anyway?
typedef struct {
  fixed_t    height;
  int        picnum;
  int        lightlevel;
  int        minx;
  int        maxx;
  // 4 padding bytes
  byte       top[SCREENWIDTH];
  byte       bottom[SCREENWIDTH];
} visplane_t;

// Here comes the obnoxious "visplane".
#define MAXVISPLANES    128
visplane_t   visplanes[MAXVISPLANES];
visplane_t*  lastvisplane;
```

A visplane can describe a horizontally **non-contiguous** region (e.g., a red floor visible in three separate windows is one visplane with three column ranges).

### Generation (during wall rendering)

For each wall/portal fragment, two potential visplanes exist:

- **Above the segment** — between the segment's top and the previous ceiling occlusion → ceiling visplane.
- **Below the segment** — between the segment's bottom and the previous floor occlusion → floor visplane.

Walking the same 3-sector room from the vertical clipping example, in render order:

- **A**: full-height wall, no gaps → no visplanes.
- **C, D**: leave gaps top and bottom → 2 visplanes per fragment (e.g., `1`,`2`,`3`,`4`).
- **B** (portal): gaps above its upper texture and below its lower texture → visplanes `5`,`6`.
- **E** (wall in subsector 2): gaps above E and below E, where the boundaries come from the *current* `ceilingclip`/`floorclip` (which were advanced by B's upper/lower textures) → visplanes `7`,`8`.
- **F** (portal where the adjacent sector opens up, so no upper/lower texture renders): nothing is drawn, but F's middle-region screen coordinates are still computed and used to emit visplanes `9`,`10` — the regions of sector 3's ceiling and floor visible through F.
- **G, H**: visplanes `11`,`12`,`13`,`14`.

### Merging

Naively this generates too many visplanes. DOOM merges where possible. A new visplane request can be absorbed into an existing one when **all of these hold**:

- Same height.
- Same picnum.
- Same lightlevel.
- For every screen-space `X` in the new request, the existing visplane is **unused at that column** — i.e., the new columns extend the visplane sideways without conflicting with already-recorded `top`/`bottom` values. Even one pixel of conflict at a shared `X` blocks the merge.

In practice this means the new request must be side-by-side with the existing visplane (or non-overlapping). When eligible, merging just extends the existing entry's column coverage; no new `visplane_t` slot is consumed. `R_FindPlane` searches linearly for an existing match:

```c
visplane_t* R_FindPlane(fixed_t height, int picnum, int lightlevel) {
    visplane_t*    check;
    ...
    for (check=visplanes; check<lastvisplane; check++) {
        if (height == check->height &&
            picnum == check->picnum &&
            lightlevel == check->lightlevel)
                break;
     }

    if (check < lastvisplane)
        return check;

    if (lastvisplane - visplanes == MAXVISPLANES)
        I_Error ("R_FindPlane: no more visplanes");

    lastvisplane++;
    check->height = height;
    check->picnum = picnum;
    check->lightlevel = lightlevel;
    check->minx = SCREENWIDTH;
    check->maxx = -1;
    memset (check->top,0xff,sizeof(check->top));

    return check;
}
```

(Mergeability — the side-by-side test — is checked elsewhere when actually adding columns to the chosen visplane.)

For complex E1M1 frames: 179 visplanes without merging, 28 with merging.

### Sizing and the visplane overflow

- `sizeof(visplane_t)` ≈ 664 bytes.
- 128 visplanes × 664 = **84,992 bytes** — about 2% of DOOM's 4 MiB minimum.
- If `lastvisplane` reaches the cap, the engine calls `I_Error`, terminates, and dumps the player back to DOS:

```text

C:\DOOM>R_FindPlane: no more visplanes

```

This crash terrified map designers because the cause was non-obvious. (Lee Killough later replaced the linear search with an O(1) chained hash table and lifted the limit; see *"The Truth about Visplane Overflows"*.)

The 128-cap was both a memory budget and a runtime budget — `R_FindPlane`'s linear scan was already O(n) and would have been a bottleneck if `n` grew.

### Drawing the flats

```c
void R_DrawPlanes (void) {
  visplane_t  *pl;
  int         light;
  int         x, stop;
  int         angle;

  for (pl = visplanes ; pl < lastvisplane ; pl++) {

    if (pl->minx > pl->maxx)
      continue;

    // sky flat
    [...] // Special case where perspective is disabled.

    // regular flat
    ds_source = W_CacheLumpNum(firstflat + flattranslation[pl->picnum],PU_STATIC);
    planeheight = abs(pl->height-viewz);
    light = (pl->lightlevel >> LIGHTSEGSHIFT)+extralight;
    planezlight = zlight[light];

    pl->top[pl->maxx+1] = 0xff;
    pl->top[pl->minx-1] = 0xff;

    stop = pl->maxx + 1;
    for (x=pl->minx ; x<= stop ; x++)
      R_MakeSpans (x,pl->top[x-1],pl->bottom[x-1] ,
      pl->top[x],pl->bottom[x]);

    Z_ChangeTag (ds_source, PU_CACHE);
  }
}
```

Even though visplanes are stored as columns, they are **rendered as horizontal spans** — `R_MakeSpans` walks neighboring columns and emits horizontal runs whenever `top`/`bottom` change. Spans give:

1. Constant `z` along the run ⇒ fast perspective-correct texturing.
2. Constant lightmap along the run ⇒ amortized lighting selection (see below).

The texture lump is fetched via the cached resource manager (`W_CacheLumpNum`) and demoted to `PU_CACHE` after use so it can be evicted later.

**Sky** is special-cased: a sector with the special "sky" ceiling number gets perspective disabled, height set to 0, and lightmap forced to 0. Each visible sky column is drawn via `colfunc` as a single column of pixels.

## Diminishing lighting (COLORMAP)

### Goal

The art direction (Aliens-inspired claustrophobia) required colors fading to black with distance, plus per-sector dim/bright lighting controlled by map designers.

### Constraint

VGA mode 13h gives a 256-color palette. A naive 16-color × 16-shade scheme would have crippled artists.

### Trick: indirection table

The `COLORMAP` lump is a 256×33 table of palette indices:

- 256 rows — one per palette color.
- 32 columns — light levels from `0` (brightest, identity) to `31` (darkest, all near-black).
- Each row is a 32-step gradient from the original color toward black, expressed using *only* existing palette indices.
- A 33rd lightmap exists: a 256-entry mapping to grays, used during invulnerability.

To render a lit pixel: given texel value `T ∈ [0,255]` and light level `L ∈ [0,31]`, look up `COLORMAP[L][T]` and write that to the framebuffer. Cost: one indirect byte fetch.

### Selecting the light level

$$\text{lightmapId} = \text{sectorLightLevel} + z \cdot \text{diminishingFactor}$$

$$\text{color} = \text{COLORMAP}[\text{lightmapId}][\text{textureTexel}]$$

Because the formula is per-`z` and per-sector, **the per-pixel cost would be high if computed naively**. The renderer caches selected lightmap IDs per screen-space scanline and per sector ID, exploiting the rendering layout (constant `z` along columns for walls, along spans for flats) so the lightmap is selected once per primitive.

### Wall orientation embellishment

To fake directional lighting:

- North-south walls: lightmap index −1 (one step **brighter**).
- East-west walls: lightmap index +1 (one step **darker**).
- Other orientations: unchanged.

### Sky bug

Skies bypass diminished lighting (no `COLORMAP` indirection, perspective also off). When the player picks up the invulnerability powerup — which switches everything to the gray-shade lightmap — the sky stays its normal colors. Outdoors, this produces a striking inconsistency: gray world, colored sky.

iOS / OpenGL ES 1.0 ports approximated invulnerability with `glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO)`, with mixed results.

## Masked rendering

"Masked" elements are everything not in the BSP-driven environment pass: monsters, items, decorations, the player's weapon, and **partially-transparent walls** (e.g., grates). Drawn last, **back-to-front**, because transparency composites correctly that way only.

```c
void R_DrawMasked (void){
  vissprite_t  *spr;
  drawseg_t    *ds;

  R_SortVisSprites ();

  // draw all vissprites back to front
  if (vissprite_p > vissprites) {
    for (spr= vsprsortedhead.next ; spr != &vsprsortedhead;
         spr = spr->next)
      R_DrawSprite (spr);
  }

  // render any remaining masked mid textures
  for (ds=ds_p-1 ; ds >= drawsegs ; ds--)
    if (ds->maskedtexturecol)
      R_RenderMaskedSegRange (ds, ds->x1, ds->x2);

  // draw the psprites on top of everything
  if (!viewangleoffset)   // don't draw on side views
    R_DrawPlayerSprites ();
}
```

### Three working data structures

#### 1. `vissprites[]` — visible things

Built during BSP traversal. Each subsector enumerates its things and adds entries:

```c
typedef struct vissprite_s {
  struct vissprite_s*  prev; // Doubly linked list.
  struct vissprite_s*  next;
  int    x1;
  int    x2;
  fixed_t  scale;
  int      patch;
  lighttable_t*  colormap;
} vissprite_t;

#define MAXVISSPRITES   128
vissprite_t   vissprites[MAXVISSPRITES];
vissprite_t*  vissprite_p;
vissprite_t   vsprsortedhead;
```

Things are added in BSP traversal order, which is *pseudo*-ordered — close enough that some sorting work is saved, but not usable as-is. `R_SortVisSprites` produces the correct back-to-front order by **updating the doubly-linked list pointers only**; array entries are never copied.

#### 2. `drawsegs[]` — log of occluders

Built during wall rendering. It records every screen-affecting segment so masked rendering can replay it for clipping.

```c
typedef struct drawseg_s {
  seg_t   *curline;
  int     x1, x2;
  fixed_t scale1, scale2, scalestep;
  int     silhouette;   // 0=none, 1=bottom, 2=top, 3=both
  fixed_t bsilheight;   // don't clip sprites above this
  fixed_t tsilheight;   // don't clip sprites below this
  // pointers to lists for sprite clipping
  short *sprtopclip;  // adjusted so [x1] is first value
  short *sprbottomclip; // adjusted so [x1] is first value
  short *maskedtexturecol; // adjusted so [x1] is first value
} drawseg_t;

#define  MAXDRAWSEGS    256
drawseg_t  drawsegs[MAXDRAWSEGS];
```

Entries are added:

- **One per wall** that emitted pixels.
- **Up to two per portal** — one for the upper part, one for the lower (if no middle texture).
- **One per masked segment** that was deferred (those need rendering during the masked pass).

Entries are naturally **ordered by distance** because they were added during the front-to-back wall pass. Distance is represented as **`scale`**, not raw `z` — this is the projection scale, easier to compare against vissprite scales and free since it was already computed during projection.

#### 3. `openings[]` — clipping pool

```c
#define MAXOPENINGS    SCREENWIDTH*64
short                  openings[MAXOPENINGS];
short*                 lastopening;
```

A shared pool of `short` arrays. Each `drawseg_t`'s `sprtopclip`, `sprbottomclip`, and `maskedtexturecol` pointers point into `openings[]`. `lastopening` is the bump-allocator pointer; clearing the pool at frame start is just `lastopening = openings`.

### `R_DrawMasked` algorithm

1. **Sort** vissprites back-to-front via `R_SortVisSprites`. Only `prev`/`next` pointers move; array entries stay put.
2. **For each sprite, back-to-front**, call `R_DrawSprite`:
   - Linearly scan `drawsegs[]` for entries whose `scale` indicates they are **in front of** the sprite, intersecting its `x1..x2` range.
   - Build a per-column occlusion rectangle from those drawsegs' `sprtopclip`/`sprbottomclip`.
   - Clip the sprite's columns against that rectangle and render.
   - **Hack:** during this scan, if a drawseg has `maskedtexturecol != NULL`, render it inline via `R_RenderMaskedSegRange`. This is how transparent walls get correctly interleaved with sprites — the sprite-pass scan triggers them at the right depth.
3. **Tail pass**: any masked segments not consumed during the sprite pass (they were behind every sprite, or there were no sprites) are rendered now by walking `drawsegs` in reverse.
4. **Player weapon**: `R_DrawPlayerSprites` paints the `psprite` on top with no clipping. Only honored if `viewangleoffset == 0` (a leftover gate from the removed three-screen mode — see below).

### Why this two-pass structure exists

Sprites and transparent walls cannot trivially interleave with environment because environment is drawn front-to-back (zero overdraw) and masked is drawn back-to-front. The drawseg log + scale-based replay is the cheap proxy for a true depth buffer (which the era's RAM and CPU couldn't afford).

### Three-screen mode artifact

DOOM up through v1.2 supported a wide-FOV setup using three networked PCs each rendering a third of the view. The CLI to enable it was removed in later versions, but the feature itself remains — that's why `R_DrawMasked` checks `viewangleoffset`: side views shouldn't show the player's weapon. Chocolate DOOM re-enabled this mode and added a single-machine variant. (Carmack reportedly designed it after visiting an Alaska Airlines flight simulator.)

## Picture format (sprites and column-based textures)

DOOM textures and sprites are stored **rotated 90° counter-clockwise** so that one screen-column's bytes are contiguous in memory — the i486 cacheline layout the renderer wants when drawing column-by-column.

Each lump is a collection of **posts** (columns). A post is a sequence of **spans**:

```text
post = [span]+ followed by terminator
span = (yOffset: u8, length: u8, payload[length]: u8)
```

This gives free transparency: the gap between the end of one span and the start of the next is implicitly transparent.

Encoding overhead is 2 bytes per span. For sparse columns (typical sprite edges) compression is good; for dense columns it can be a slight loss.

**Lost Soul sprite (44×47, 2068 pixels):**

- Row 44: a single span of 2 texels = 4 bytes.
- Row 33: two spans = 48 bytes (rare loss vs. uncompressed 47 bytes).
- Row 10: two spans = 19 bytes (vs. 47 uncompressed — ~60% saved).
- Whole sprite: 1360 bytes — ~50% compression overall.

## Sprite aspect ratio

The VGA framebuffer is 320×200. CRT monitors of the era stretched this to ~4:3, making **pixels taller than they are wide**. Artists working in Deluxe Paint at 320×200 saw non-square pixels, so their work was authored for the stretched output.

Sprites are stored circular but rendered elliptical — the round-on-disk vs. tall-on-screen difference is exactly the CRT correction. Asset extractors that ignored this produced bulky monsters; even some merchandise (a Reaper Miniatures Cacodemon figurine) made the same mistake.

## Glossary

- **BAM** — Binary Angular Measurement. Angle as a 32-bit unsigned integer where `0x00000000`=0°, `0x40000000`=90°, etc.
- **BSP** — Binary Space Partition. Precomputed tree that partitions the 2D map into convex subsectors and provides constant-cost front-to-back ordering.
- **colfunc** — function pointer for column drawing; variants for opaque, masked, fuzz (spectres), translated (player colors), etc.
- **COLORMAP** — 256×33 palette-indirection table for diminishing lighting; row=color, column=light level (0 brightest, 31 darkest, 32 = invulnerability gray).
- **drawseg** — log entry for an environment segment that emitted pixels; carries scale + clipping pointers used by `R_DrawSprite` to occlude masked elements.
- **flat** — a horizontal floor or ceiling texture; rendered via visplanes as horizontal spans.
- **fragment** — a segment after horizontal clipping; may be a piece of an original segment.
- **lightmap / lightmapId** — index into a row of `COLORMAP`; 0 = brightest, 31 = darkest.
- **masked** — anything drawn back-to-front in the final pass: sprites, transparent walls, the player's weapon (`psprite`).
- **node** — internal BSP node holding a partition line `(x, y, dx, dy)` plus child bounding boxes.
- **picnum** — texture identifier.
- **portal** — a two-sided segment connecting two sectors; usually transparent in its middle, may have upper/lower textures for height differences.
- **post** — a column of pixels in the picture format; a sequence of spans.
- **psprite** — player sprite; the weapon HUD overlay.
- **scale** — the projection scale of a column/segment; used as a fast distance proxy in clipping logic.
- **sector** — a logical floor/ceiling region defined by the level designer (one floor height, one ceiling height, one floor texture, etc.).
- **seg / segment** — a piece of a line after BSP construction; basic unit of wall rendering.
- **solidsegs** — array of horizontal screen ranges currently occluded; used by the crude horizontal clipping pass.
- **span** — *Two distinct meanings.* (1) In the picture format: a contiguous run of opaque texels in a column. (2) In flat rendering: a horizontal run of pixels of a single visplane on a single scanline.
- **subsector (SSECTOR)** — convex region produced by BSP splitting; contains a list of segs.
- **vissprite** — visible thing entry queued during BSP traversal for the masked pass.
- **visplane** — screen-space description of a floor or ceiling region: height, picnum, lightlevel, and per-column top/bottom extents.
- **wall** — a single-sided opaque segment with only a middle texture.
