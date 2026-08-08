"""Shared Modal image for torchwright_doom (used by modal_test.py and
modal_run.py).

Path resolution assumes the umbrella structure (torchwright_doom and
torchwright are sibling submodules under torchdoom). Standalone clones
of torchwright_doom can't run Modal entrypoints without the sibling
torchwright checkout — Modal is workspace-mode-only.
"""

import importlib.util
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
# Resolve the sibling torchwright checkout through the installed (editable)
# package rather than directory layout — the layout guess breaks in git
# worktrees, where this file is not directly under the umbrella.
_TW_SPEC = importlib.util.find_spec("torchwright")
if _TW_SPEC is None or _TW_SPEC.origin is None:
    raise ImportError(
        "torchwright is not importable — Modal entrypoints need the umbrella "
        "workspace venv (see the module docstring)"
    )
_TORCHWRIGHT = Path(_TW_SPEC.origin).resolve().parents[1]

_BASE_IMAGE = modal.Image.debian_slim(python_version="3.12").uv_sync(
    groups=["dev"], extra_options="--no-install-project"
)

_CONFIGS = _HERE / "configs"

_IGNORE_PARTS = {"__pycache__", "token_dumps", "specs", "scripts", ".git", ".venv"}


def _ignore(p: Path) -> bool:
    return any(part in _IGNORE_PARTS for part in p.parts) or p.suffix == ".pyc"


def _with_workspace_sources(image: modal.Image) -> modal.Image:
    """Attach local files last, after every dependency-building operation."""
    return (
        image.add_local_file(str(_HERE / "doom1.wad"), "/root/doom1.wad")
        .add_local_python_source(
            "torchwright", "torchwright_doom", "tests", "scripts", "modal_image"
        )
    )


def _with_assets(image: modal.Image) -> modal.Image:
    return (
        _with_workspace_sources(image)
        .add_local_dir(str(_CONFIGS), "/root/configs", ignore=_ignore)
        .add_local_file(str(_HERE / "doom1.wad"), "/root/configs/doom1.wad")
    )


# Generic script image; production assets add numba but not ONNX Runtime.
IMAGE = _with_workspace_sources(_BASE_IMAGE)
ASSETS_IMAGE = _with_assets(
    _BASE_IMAGE.uv_pip_install("numba").env(
        {
            "HF_ENABLE_PARALLEL_LOADING": "true",
            "HF_PARALLEL_LOADING_WORKERS": "8",
        }
    )
)

# ONNX Runtime exists only in the diagnostic/full-test image. The production
# compile/render image above has no ORT dependency or import path.
ONNX_DIAGNOSTIC_IMAGE = _with_assets(
    _BASE_IMAGE.uv_pip_install("numba", "onnxruntime-gpu==1.26.0")
)
TEST_IMAGE = ONNX_DIAGNOSTIC_IMAGE

# Production direct-HF bundles. Kept separate from the legacy/diagnostic ONNX
# volume so one backend can never satisfy the other's cache probe.
HF_BUNDLE_VOLUME = modal.Volume.from_name(
    "torchwright-doom-hf-phi3", create_if_missing=True
)

# ONNX is diagnostic-only (debug sessions and backend investigations).
ONNX_DEBUG_VOLUME = modal.Volume.from_name(
    "torchwright-doom-render-cache", create_if_missing=True
)
