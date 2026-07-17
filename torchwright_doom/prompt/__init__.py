"""The prefill pipeline — turn a DOOM WAD map into the token sequence the
transformer reads before autoregression begins.

Dataflow (one direction, top to bottom):

- ``types`` — the :class:`MapData` schema (dense vertex / linedef / sidedef /
  sector / subsector / seg / node indices) + per-frame :class:`GameState`.
  A raw WAD-loaded ``MapData`` is integer-coord with ``scene_origin == (0, 0)``;
  the subset step renumbers and mean-centres it.
- ``wad`` — parse the seven geometry lumps (plus ``THINGS``) of a WAD into a
  raw ``MapData``. Texture *names* only; no pixels.
- ``subset`` — :func:`subset_by_bbox`: keep the segs/subsectors and minimal BSP
  subtree inside a world-space box, renumber to dense indices, mean-centre the
  coordinates, and store the centroid in ``scene_origin``.
- ``geometry`` — :func:`bake_segments`: walk seg -> linedef -> sidedef -> sector
  once to resolve each seg's endpoints, heights, and texture names (a baked
  :class:`Segment`, distinct from the raw ``MapData.segs`` entry).
- ``plane_tables`` — :func:`build_plane_tables`: dedup floors/ceilings into a
  stable visplane list and tag each subsector with its floor/ceiling plane id.
- ``build`` — :func:`build_prompt`: emit the flat ``list[Token]`` prefill
  (player state -> per-node -> per-subsector/seg -> visplane defs -> ``BEGIN``),
  in the ``PROTOCOL.md`` prefill order.
- ``scene`` — the production entry point: :func:`load_render_scene` (WAD +
  config region + asset book), :func:`pose_from_world` (world pose into the
  subset frame), and :func:`prefill_rows_for` (prompt row ids, via
  ``tokenizer.rows``).
- ``scenes`` — a :class:`Scene` (WAD path, subset box, initial pose) and
  :func:`load`, which opens the WAD, subsets it, and shifts the pose into the
  subset frame. The test-fixture entry point.

The production entry point is ``prompt.scene.prefill_rows_for``.
This package re-exports nothing; ``doom_sandbox`` is never imported here.
"""
