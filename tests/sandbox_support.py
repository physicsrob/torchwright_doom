"""Shared scaffolding for the cross-submodule oracle/bridge gates.

``doom_sandbox`` is a ``package = false`` workspace sibling: it is never
pip-installed, so the gates import it as a directory package from the
checkout that contains it.  Two layouts exist:

- the umbrella checkout: this file is
  ``<umbrella>/torchwright_doom/tests/sandbox_support.py`` and the sibling
  is at ``<umbrella>/doom_sandbox`` — two levels up;
- the Modal test container: ``add_local_python_source`` lands sources flat
  under ``/root`` (``/root/tests``, ``/root/doom_sandbox``) — one level up.

``TWDOOM_UMBRELLA`` overrides the probe explicitly.

In a genuine standalone checkout (no sibling anywhere) the gates skip.
When ``TWDOOM_REQUIRE_SANDBOX_GATES`` is set — the Modal test entrypoint
sets it — a missing or import-broken sandbox FAILS instead of skipping,
so the oracle gates cannot silently fall out of CI (audit finding #0: the
gates skipped on every ``make test`` run while the suite stayed green).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

_SKIP_REASON = "doom_sandbox sibling not present (standalone checkout)"


def _gates_required() -> bool:
    return bool(os.environ.get("TWDOOM_REQUIRE_SANDBOX_GATES"))


def umbrella_root() -> Path | None:
    """The directory containing the ``doom_sandbox`` sibling, or ``None``.

    Pure probe — touches neither ``sys.path`` nor the environment.  A set
    ``TWDOOM_UMBRELLA`` is authoritative: no fallback probing, so a wrong
    value fails loud rather than being papered over.
    """
    env = os.environ.get("TWDOOM_UMBRELLA")
    here = Path(__file__).resolve()
    candidates = [Path(env)] if env else [here.parents[2], here.parents[1]]
    for candidate in candidates:
        if (candidate / "doom_sandbox").is_dir():
            return candidate
    return None


def require_doom_sandbox() -> Path:
    """The gate preamble (previously copy-pasted into each gate file):
    disable the numba JIT, locate the sibling, put its parent on
    ``sys.path``, and return that parent — the umbrella root, from which
    ``test_schema_sync`` reads the committed vocab-contract baseline.

    Skips in a standalone checkout; fails loud under
    ``TWDOOM_REQUIRE_SANDBOX_GATES``.
    """
    # Must be set before the first sandbox import: the implementation
    # modules bind numba at import time, and the gates run JIT-disabled.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    root = umbrella_root()
    if root is not None and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import doom_sandbox
    except ImportError as exc:
        if _gates_required():
            pytest.fail(
                f"TWDOOM_REQUIRE_SANDBOX_GATES is set but doom_sandbox cannot "
                f"be imported ({exc}) — the oracle gates would silently skip "
                f"in an environment that requires them"
            )
        pytest.skip(_SKIP_REASON)
    return Path(doom_sandbox.__file__).resolve().parent.parent


def import_sandbox(name: str):
    """``pytest.importorskip`` for the gates' cross-submodule imports
    (``doom_sandbox.*``, ``scripts.vocab_diff``) that fails loud under
    ``TWDOOM_REQUIRE_SANDBOX_GATES`` — in CI a broken gate import is a
    broken gate, not a skippable environment.
    """
    if _gates_required():
        return importlib.import_module(name)
    return pytest.importorskip(name)
