"""The shipped DOOM tokenizer file is import-clean.

``tokenizer/tokenization_doom.py`` is copied verbatim into every saved tokenizer
directory (by ``transformers``' ``custom_object_save``) and must load on a
machine with **only ``transformers``** — no ``torch``, no ``torchwright_doom``.
That holds iff it imports only the standard library + ``transformers`` and has no
relative imports.

A stray ``import torch`` / ``from ..x`` would pass every *in-package* test (where
torch is present) and fail only on the stranger's machine. This catches it
statically — mirroring the import scan ``transformers`` itself runs at load time
— and needs no renderer env or trace, so it runs in cloud CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SUBMODULE_ROOT = Path(__file__).resolve().parents[2]
_SHIPPED_SOURCE = (
    _SUBMODULE_ROOT / "torchwright_doom" / "tokenizer" / "tokenization_doom.py"
)

pytest.importorskip("transformers")


def test_shipped_file_imports_only_stdlib_and_transformers() -> None:
    from transformers.dynamic_module_utils import get_imports, get_relative_imports

    assert (
        get_relative_imports(str(_SHIPPED_SOURCE)) == []
    ), "shipped file has relative imports — it must be standalone"
    allowed = set(sys.stdlib_module_names) | {"transformers"}
    leaked = set(get_imports(str(_SHIPPED_SOURCE))) - allowed
    assert (
        not leaked
    ), f"shipped file imports non-stdlib/non-transformers: {sorted(leaked)}"
