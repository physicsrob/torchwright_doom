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
    # onnxruntime-gpu is PINNED: the CUDA-graph capture mechanics the render
    # runtime relies on (per-annotation-id run counters, gpu_graph_id "-1"
    # skip semantics, capture-on-3rd-run) are version-specific and were
    # verified against the 1.26.0 source.  Keep in lockstep with the local
    # venv (plan_cuda_graph_decode.md, step 0).
    .uv_pip_install("numba", "onnxruntime-gpu==1.26.0")
    .add_local_file(str(_TORCHWRIGHT / "E8.8.1024.txt"), "/root/E8.8.1024.txt")
    .add_local_file(str(_HERE / "doom1.wad"), "/root/doom1.wad")
    .add_local_python_source(
        "torchwright", "torchwright_doom", "tests", "scripts", "modal_image"
    )
)
