"""Make the sibling ``doom_sandbox`` checkout importable for render tests.

``doom_sandbox`` is not pip-installed into the torchwright_doom test venv; it's
imported as a directory package from the umbrella checkout (same convention as
``tests/scene/test_flat_pixel_oracle.py``). Inserting the umbrella root on
``sys.path`` at collection time lets ``pytest.importorskip("doom_sandbox...")``
find it; in a standalone checkout (no sibling) the importorskips skip cleanly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[3]

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
if (_UMBRELLA / "doom_sandbox").is_dir() and str(_UMBRELLA) not in sys.path:
    sys.path.insert(0, str(_UMBRELLA))
