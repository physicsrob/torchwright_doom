"""Plan K render on Modal A100-80GB, with artifact sync-back.

    uv run modal run modal_render.py --fixture e1m1_subset_textured --pose 0 --mode pure_ar
    uv run modal run modal_render.py --fixture e1m1_subset_textured --pose 0 --mode both

The full frame's KV footprint is too large for the local L4, so generation runs
on an A100-80GB. Artifacts (generated/reference/diff PNGs + token_dump.json) are
written to a ``modal.Volume`` (durable; inspect later with ``modal volume get``)
*and* returned so the local entrypoint mirrors them to ``out/<run>/`` on disk —
``make modal-run`` only captures stdout, which is exactly why this dedicated
entrypoint exists (the one sanctioned new root ``modal_*.py``).

The image extends the shared ``IMAGE`` with ``numba`` (a ``doom_sandbox`` runtime
dep not in torchwright_doom's) and mounts the sibling ``doom_sandbox`` checkout
(code + fixture JSONs + WAD) at ``/root/doom_sandbox`` so the reference renderer
and drafter import there.
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

from modal_image import IMAGE

_HERE = Path(__file__).resolve().parent
_DOOM_SANDBOX = _HERE.parent / "doom_sandbox"

_IGNORE_PARTS = {"__pycache__", "token_dumps", "specs", "scripts", ".git", ".venv"}


def _ignore(p: Path) -> bool:
    return any(part in _IGNORE_PARTS for part in p.parts) or p.suffix == ".pyc"


RENDER_IMAGE = IMAGE.pip_install("numba").add_local_dir(
    str(_DOOM_SANDBOX), "/root/doom_sandbox", ignore=_ignore
)

app = modal.App("torchwright-doom-render", image=RENDER_IMAGE)
RENDER_VOLUME = modal.Volume.from_name(
    "torchwright-doom-render", create_if_missing=True
)


@app.function(
    gpu="a100-80gb",
    cpu=8,
    memory=65536,
    timeout=5400,
    volumes={"/artifacts": RENDER_VOLUME},
)
def render_remote(run_id: str, kwargs: dict) -> dict:
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")  # make /root/doom_sandbox importable

    from torchwright_doom.render.cli import run_render

    out_dir = f"/artifacts/{run_id}"
    summary = run_render(out_dir=out_dir, **kwargs)
    RENDER_VOLUME.commit()

    files = {
        p.name: p.read_bytes()
        for p in sorted(Path(out_dir).glob("*"))
        if p.is_file()
    }
    return {"summary": summary, "files": files}


@app.local_entrypoint()
def main(
    fixture: str = "e1m1_subset_textured",
    pose: int = 0,
    mode: str = "pure_ar",
    run_name: str = "",
    max_positions: int = 8000,
    d: int = 4096,
    d_head: int = 32,
    scale: int = 8,
    draft_window: int = 8,
):
    run_id = run_name or f"{fixture}__pose{pose}__{mode}__{int(time.time())}"
    kwargs = dict(
        fixture=fixture,
        pose_index=pose,
        mode=mode,
        max_positions=max_positions,
        d=d,
        d_head=d_head,
        scale=scale,
        draft_window=draft_window,
    )
    print(f"[local] launching render_remote run_id={run_id} kwargs={kwargs}")
    result = render_remote.remote(run_id, kwargs)

    local_dir = _HERE / "out" / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    for name, data in result["files"].items():
        (local_dir / name).write_bytes(data)
    print("\n" + result["summary"]["report_text"])
    print(f"\n[local] artifacts -> {local_dir}")
    print(f"[local] files: {sorted(result['files'])}")
    if result["summary"].get("footprint"):
        print(f"[local] footprint: {result['summary']['footprint']}")
