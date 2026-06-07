"""Plan K render harness — run the *compiled* DOOM transformer autoregressively
to *generate* a frame, decode the generated token stream to pixels, and compare
against the reference render.

Submodules:
- ``tokens_bridge`` — token <-> W_EMBED-row encode/decode + sandbox name bridge.
- ``compiled_model`` — build + compile the token-id ``forward`` (the artifact).
- ``pure_ar`` — pure autoregressive rollout over ``compiled.step`` (Step 1).
- ``spec_decode`` — speculative decoding driven by the reference drafter (Step 2),
  a strict optimization that is bit-identical to ``pure_ar``.
- ``decode`` — generated token stream -> screen pixel buffer (dumb host).
- ``compare`` — fetch the reference render + report image-level agreement stats.
- ``artifacts`` — write ``token_dump.json`` + the PNGs.
- ``diagnostic`` — teacher-forced + ``probe_compiled`` divergence localizer.

Everything except the CLI is a side-effect-free library of pure functions; all
``doom_sandbox`` imports are lazy so ``torchwright_doom`` stays importable in a
standalone checkout.
"""
