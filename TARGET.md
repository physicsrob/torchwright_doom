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

**Vocab contract reference: `doom_sandbox` main.** The real
`torchwright_doom.embedding.TOKEN_VOCAB` mirrors the sandbox
`doom_sandbox.implementation.setup.VOCAB` token-for-token (108 types,
cardinality 93,114). `scripts/vocab_diff.py` (umbrella) computes the
live contract diff; the committed `baseline_vocab_diff.txt` is **empty**
and `test_schema_sync` pins live == committed. When the sandbox pin
moves, re-bump the umbrella's `doom_sandbox` pointer to main and
regenerate the baseline in one reviewable commit.

**Baseline SHAs (Plan A landing):**
- `doom_sandbox`: `23ca589` (main) — the contract reference.
- `torchwright_doom`: the Plan A commit series (`A0`…`A8`).
- umbrella: `9c0ec42` + the Plan A pointer bump (deferred; working
  locally per the current task — not yet pushed).

**Asset coupling.** The prefill contract needs the compiled-asset
surface (flat/texture counts, name→id maps, `PLAYPAL`); the pure slice
the prefill side needs is ported in `torchwright_doom/asset_config.py`.
The forward-path pixel/dimension banks remain lookup-track-owned.
