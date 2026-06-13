# Port target (Plan A: vocab alignment)

**Screen scale: 160×100 (shipped).** The spec09 port ran the whole way
at the sandbox fixture default of 60×50 (exact token-by-token prompt
parity with the sandbox fixtures); the retarget to the real **160×100**
has since shipped. The mechanism is config-driven, not a constants
edit: `configs/e1m1.yaml` sets `model.scale: 2` and
`apply_screen_env` (`inference/config.py`) exports the screen dims
before graph modules import. `torchwright_doom/constants.py` keeps
60×50 only as its bare-import default (what tests that import the graph
without a config see). Every screen-derived width references the
constants module, never a literal — guarded by
`test_screen_scale_guard` — so the scale stays a config swap, not a
re-port.

**Vocab contract.** `torchwright_doom.embedding.TOKEN_VOCAB` is the single
source of truth for the token vocabulary (108 types; cardinality 93,114 at the
60×50 fixture scale). It was proven token-for-token identical to the original
`doom_sandbox` vocab before the sandbox was removed. The vendored drafter and
the compiled model both derive their tokens from this one vocab, so they stay
token-identical by construction; the end-to-end guard is the whole-frame
routing gate (`tests/scene/test_flat_pixel_oracle.py`).

**Baseline SHAs (Plan A landing):**
- `doom_sandbox`: `23ca589` (main) — the contract reference.
- `torchwright_doom`: the Plan A commit series (`A0`…`A8`).
- umbrella: `9c0ec42` + the Plan A pointer bump (deferred; working
  locally per the current task — not yet pushed).

**Asset coupling.** The prefill contract needs the compiled-asset
surface (flat/texture counts, name→id maps, `PLAYPAL`); the pure slice
the prefill side needs is ported in `torchwright_doom/asset_config.py`.
The forward-path pixel/dimension banks remain lookup-track-owned.
