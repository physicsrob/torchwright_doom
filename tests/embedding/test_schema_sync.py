"""Plan A / A8: schema-sync.

The *live* contract diff (``scripts/vocab_diff.generate()`` over the real
``TOKEN_VOCAB`` vs the sandbox ``VOCAB``) must equal the committed
``baseline_vocab_diff.txt`` and report an empty contract — so submodule
pin drift can't silently change the real-vs-sandbox token contract.

Cross-submodule: ``doom_sandbox`` is a ``package = false`` workspace
sibling, so this test adds the umbrella checkout to ``sys.path`` and
``importorskip``\\ s it (skipped when the umbrella layout isn't present,
e.g. a standalone clone of this submodule).
"""

from __future__ import annotations

from ..sandbox_support import import_sandbox, require_doom_sandbox


def test_live_diff_matches_committed_baseline_and_is_empty() -> None:
    umbrella = require_doom_sandbox()

    import_sandbox("doom_sandbox.implementation.setup")
    vocab_diff = import_sandbox("scripts.vocab_diff")

    baseline_path = umbrella / "baseline_vocab_diff.txt"
    assert baseline_path.is_file(), f"missing committed baseline: {baseline_path}"

    live = vocab_diff.generate()
    committed = baseline_path.read_text()
    assert live == committed, (
        "live vocab contract diff != committed baseline_vocab_diff.txt — "
        "regenerate it (NUMBA_DISABLE_JIT=1 python scripts/vocab_diff.py > "
        "baseline_vocab_diff.txt) as a deliberate, reviewable change"
    )
    assert (
        "[contract diff empty] true" in live
    ), "real TOKEN_VOCAB no longer mirrors the sandbox contract"
