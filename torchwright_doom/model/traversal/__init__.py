"""The BSP walk: R_RenderBSPNode token transitions (bsp_traversal),
R_CheckBBox visibility pruning (bbox_pruning), stack-pop edges for
TRAVERSE_RETURN (traversal_edges), and the solidsegs horizontal
occlusion channel (solid_intervals).  solid_intervals lives here, not
in raster/: it is the occlusion state the walk and the seg scan query
— original DOOM keeps solidsegs in r_bsp.c too."""
