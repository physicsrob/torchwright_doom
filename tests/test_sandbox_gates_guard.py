"""Suite-level guard (audit finding #0): the cross-submodule oracle gates
must RUN under ``make test``, not skip.

Without ``TWDOOM_REQUIRE_SANDBOX_GATES`` this guard skips — a genuine
standalone checkout legitimately skips the gates.  The Modal test
entrypoint (``modal_test.py``) sets the marker, so any image or plumbing
refactor that breaks in-container sandbox discovery turns the suite RED
here instead of quietly going green-with-skips again.
"""

from __future__ import annotations

import os

import pytest

from .sandbox_support import require_doom_sandbox


def test_oracle_gates_can_run() -> None:
    if not os.environ.get("TWDOOM_REQUIRE_SANDBOX_GATES"):
        pytest.skip("guard inactive (TWDOOM_REQUIRE_SANDBOX_GATES not set)")
    umbrella = require_doom_sandbox()

    # The modules every oracle/bridge gate needs — plain imports so a missing
    # dependency fails with the real traceback, not a skip.
    import doom_sandbox.fixtures  # noqa: F401
    import doom_sandbox.implementation.prefill  # noqa: F401
    import doom_sandbox.implementation.reference  # noqa: F401
    import doom_sandbox.implementation.reference_drafter  # noqa: F401

    assert (umbrella / "baseline_vocab_diff.txt").is_file(), (
        "umbrella-root baseline_vocab_diff.txt not found — test_schema_sync "
        "cannot run its contract diff"
    )
