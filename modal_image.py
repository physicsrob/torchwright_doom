"""Shared Modal image for torchwright_doom (used by modal_test.py and
modal_run.py).

Path resolution assumes the umbrella structure (torchwright_doom and
torchwright are sibling submodules under torchdoom). Standalone clones
of torchwright_doom can't run Modal entrypoints without the sibling
torchwright checkout — Modal is workspace-mode-only.
"""

from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_TORCHWRIGHT = _HERE.parent / "torchwright"

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["dev"], extra_options="--no-install-project")
    .uv_pip_install("numba", "onnxruntime-gpu")
    .add_local_file(str(_TORCHWRIGHT / "E8.8.1024.txt"), "/root/E8.8.1024.txt")
    .add_local_file(str(_HERE / "doom1.wad"), "/root/doom1.wad")
    .add_local_python_source("torchwright", "torchwright_doom", "tests", "modal_image")
)
