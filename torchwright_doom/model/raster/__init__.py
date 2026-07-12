"""Seg projection through pixels, one deliberately chunky stage: seg
projection/scanning (seg_*), wall ranges and columns (wall_*),
visplanes and flats (visplane_*, flat_*), the final pixel pass
(pixel_dispatcher, uv_compute), the weapon sprite (psprite_renderer),
the status bar (statusbar_renderer), and the dispatch glue
(range_dispatcher, payload_router).  The segs/walls/planes/pixels
sub-split is deliberately NOT directories: those clusters import each
other bidirectionally (seg_projection owns the wall/plane/flat
subcontexts by design); the filename prefixes do the grouping."""
