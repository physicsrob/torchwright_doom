"""Version-agnostic compile+run driver for the cross-refactor forward-divergence
measurement (see ``block_forward_divergence.py``).

Run as a SUBPROCESS by the orchestrator, once per code state:

- **Baseline (2a tip)**: launched under ``python -S`` with ``TW_SNAPSHOT_PATHS``
  set to the 2a torchwright + torchwright_doom worktrees and ``TW_VENV_SITE`` to
  the venv's ``site-packages``.  ``-S`` skips ``.pth`` processing so the PEP-660
  editable finder for the HEAD torchwright is never installed; this module then
  rebuilds ``sys.path`` (snapshot first, deps appended) so ``import torchwright``
  resolves to the 2a snapshot.  The 2a ``linear_relu_linear`` builds a
  ``Linear -> ReLU -> Linear`` chain and the 2a fusion accepts
  ``skip_relu_ejecting`` -> the original 64-layer chain-mined model.
- **HEAD**: launched normally (no ``TW_SNAPSHOT_PATHS``) -> the current
  native-Block + block-aware-fusion 57-layer model.

The build/compile/run steps are identical for both; only the imported
``torchwright`` differs.  ``fuse_consecutive_linears`` is called with
``skip_relu_ejecting=True`` under a ``try/except TypeError`` so the same code
works against the 2a signature (kwarg present) and HEAD (kwarg deleted).  The
compile settings mirror the earlier Gate-C divergence run exactly
(``optimize=0``, ``rms_norm_const_exp=63``) and the
prefill uses the same deterministic fallback, so the two logit tensors are
directly comparable.

Env in: ``OUT_PATH`` (where to torch.save the result), ``ONNX_PATH``, ``CONFIG``,
``D``, ``D_HEAD``, ``D_ROT``, ``D_HIDDEN``, ``MAX_SEQ_LEN``, ``PREFILL``, and
(baseline only) ``TW_SNAPSHOT_PATHS`` / ``TW_VENV_SITE``.
Saves ``{logits, tw_file, twd_file, n_layers, tokens}``.
"""

import os
import sys

# --- import bootstrap: must run BEFORE importing torch / torchwright ---------
_SNAP = os.environ.get("TW_SNAPSHOT_PATHS")
if _SNAP:
    # Snapshot dirs first so `import torchwright[_doom]` resolves to the 2a tip.
    for _p in reversed(_SNAP.split(os.pathsep)):
        if _p:
            sys.path.insert(0, _p)
    _site = os.environ.get("TW_VENV_SITE")
    if _site:
        sys.path.append(_site)  # deps (torch, onnx, ortools, ...) from the venv


def _resolve_config(config_name: str) -> str:
    from pathlib import Path

    for cand in (
        Path("/root/configs") / f"{config_name}.yaml",
        Path(__file__).resolve().parents[1] / "configs" / f"{config_name}.yaml",
    ):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(f"config {config_name}.yaml not found")


def _resolve_wad(cfg):
    from pathlib import Path

    from torchwright_doom.inference.config import resolve_wad_path

    try:
        return resolve_wad_path(cfg, base_dir=str(Path(__file__).resolve().parents[1]))
    except Exception:
        for cand in (
            Path("/root/doom1.wad"),
            Path(__file__).resolve().parents[1] / "doom1.wad",
            Path.home() / "Downloads" / "doom1.wad",
        ):
            if cand.exists():
                return str(cand)
        return None


def main() -> None:
    import json
    import time
    from pathlib import Path

    import torch

    import torchwright
    import torchwright_doom

    tw_file = torchwright.__file__
    twd_file = torchwright_doom.__file__
    print(f"[driver] torchwright     = {tw_file}", flush=True)
    print(f"[driver] torchwright_doom = {twd_file}", flush=True)

    config_name = os.environ.get("CONFIG", "e1m1")
    d = int(os.environ.get("D", "8192"))
    d_head = int(os.environ.get("D_HEAD", "128"))
    d_rot = int(os.environ["D_ROT"]) if os.environ.get("D_ROT") else 64
    d_hidden = int(os.environ.get("D_HIDDEN", "16384"))
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "65536"))
    prefill = os.environ.get("PREFILL", "e1m1")
    onnx_path = os.environ["ONNX_PATH"]
    out_path = os.environ["OUT_PATH"]

    from torchwright.graph.optimize import fuse_consecutive_linears
    from torchwright_doom.inference.compiled_model import build_graph
    from torchwright_doom.inference.config import load_render_config

    cfg = load_render_config(_resolve_config(config_name))
    t0 = time.perf_counter()
    next_token, _rope, emb, _banks = build_graph(
        d_head=d_head,
        max_positions=max_seq_len,
        d_rot=d_rot,
        asset_config=cfg.asset_config(),
        wad_path=_resolve_wad(cfg),
    )
    # 2a signature accepts skip_relu_ejecting; HEAD deleted the kwarg (fusion is
    # block-aware and width-safe by construction).  Same width-safe intent.
    try:
        n_fused = fuse_consecutive_linears(
            {next_token}, verbose=False, skip_relu_ejecting=True
        )
    except TypeError:
        n_fused = fuse_consecutive_linears({next_token}, verbose=False)
    print(
        f"[driver] built graph + fused {n_fused} pairs in "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    from torchwright.compiler.export import compile_to_onnx, meta_path_for

    t0 = time.perf_counter()
    compile_to_onnx(
        next_token,
        embedding=emb,
        output_path=onnx_path,
        d=d,
        d_head=d_head,
        d_hidden=d_hidden,
        max_seq_len=max_seq_len,
        optimize=0,
        rms_norm_const_exp=63,
        verbose=False,
    )
    meta = json.loads(Path(meta_path_for(onnx_path)).read_text())
    n_layers = meta.get("n_layers")
    print(
        f"[driver] compiled to ONNX ({n_layers} layers) in "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    from torchwright_doom.inference.hf_export import _doom_bos_eos_strings

    vocab = list(meta["vocab"])
    token_to_id = {t: i for i, t in enumerate(vocab)}
    bos_str, eos_str = _doom_bos_eos_strings(meta)
    wanted = [bos_str] + list(prefill)
    missing = [t for t in wanted if t not in token_to_id]
    if missing:
        # Same deterministic fallback as the earlier Gate-C run so the token
        # sequence — hence the comparison — is identical across both states.
        fallback = [t for t in vocab if t not in (bos_str, eos_str)][:8]
        print(
            f"[driver] prefill tokens {missing!r} not in vocab; "
            f"falling back to {fallback!r}",
            flush=True,
        )
        wanted = [bos_str] + fallback
    ids = torch.tensor([[token_to_id[t] for t in wanted]], dtype=torch.int64)

    from torchwright.compiler.hf.convert import convert_onnx_to_hf

    t0 = time.perf_counter()
    model = convert_onnx_to_hf(onnx_path, bos_token=bos_str, eos_token=eos_str)
    model = model.to(torch.float32).eval()
    print(
        f"[driver] converted to HF/torch in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]
    logits = logits.detach().cpu()
    print(
        f"[driver] prefill {ids.shape[1]} tokens in {time.perf_counter() - t0:.1f}s "
        f"-> logits {tuple(logits.shape)}",
        flush=True,
    )

    torch.save(
        {
            "logits": logits,
            "tw_file": tw_file,
            "twd_file": twd_file,
            "n_layers": n_layers,
            "tokens": wanted,
        },
        out_path,
    )
    print(f"[driver] saved logits -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
