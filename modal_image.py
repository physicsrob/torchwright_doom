"""Shared Modal image for torchwright_doom modal scripts (modal_walkthrough,
modal_dump_phase_e).

Path resolution assumes the umbrella structure (torchwright_doom and
torchwright are sibling submodules under torchdoom). Standalone clones
of torchwright_doom can't run these without the sibling torchwright
checkout — Modal is workspace-mode-only.
"""

from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_TORCHWRIGHT = _HERE.parent / "torchwright"

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["dev"], extra_options="--no-install-project")
    .add_local_file(str(_TORCHWRIGHT / "E8.8.1024.txt"), "/root/E8.8.1024.txt")
    .add_local_file(str(_HERE / "doom1.wad"), "/root/doom1.wad")
    .add_local_python_source(
        "torchwright", "torchwright_doom", "tests", "scripts", "modal_image"
    )
)
