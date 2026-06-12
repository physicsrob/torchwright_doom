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

_DOOM_SANDBOX = _HERE.parent / "doom_sandbox"
_CONFIGS = _HERE / "configs"

_IGNORE_PARTS = {"__pycache__", "token_dumps", "specs", "scripts", ".git", ".venv"}


def _ignore(p: Path) -> bool:
    return any(part in _IGNORE_PARTS for part in p.parts) or p.suffix == ".pyc"


# IMAGE plus the sibling ``doom_sandbox`` checkout (reference renderer/drafter
# + fixture JSONs), the committed configs, and the WAD next to them — what the
# render and artifact-debugging entrypoints (modal_render.py, modal_run.py's
# GPU function) need beyond the python sources.
ASSETS_IMAGE = (
    IMAGE.add_local_dir(str(_DOOM_SANDBOX), "/root/doom_sandbox", ignore=_ignore)
    .add_local_dir(str(_CONFIGS), "/root/configs", ignore=_ignore)
    .add_local_file(str(_HERE / "doom1.wad"), "/root/configs/doom1.wad")
)

# Compiled-ONNX cache, keyed by the LOCALLY-computed cache key (it embeds git
# SHAs no Modal container can derive).  Shared by the compile/render flow and
# the artifact-debugging scripts.
CACHE_VOLUME = modal.Volume.from_name(
    "torchwright-doom-render-cache", create_if_missing=True
)
