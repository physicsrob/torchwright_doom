# Port target (Plan A: vocab alignment)

**Screen scale: 60×50** — the sandbox fixture default. The spec09 port
runs the whole way at 60×50, which gives exact token-by-token prompt
parity with the sandbox fixtures. The retarget to the real **160×100**
is **deferred to a project-level step once the port is in good shape**:
a one-line change to `SCREEN_WIDTH` / `SCREEN_HEIGHT` in
`torchwright_doom/constants.py`. It is kept a constant swap (not a
re-port) by the `test_screen_scale_guard` no-bare-literal test, and it
is **not** a per-plan completion gate.

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
