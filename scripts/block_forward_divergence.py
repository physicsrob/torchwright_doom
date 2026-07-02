"""Gate-C Task 1: measured flagship forward divergence, chain-mined vs blockified.

Compiles the production e1m1 graph to ONNX on BOTH paths (chain-mined and
blockified), runs the SAME token prefill through each under onnxruntime, and
reports the max absolute logit divergence and whether the per-position argmax
(the production token-level equivalence bar) agrees.

The two schedules are NOT identical (see block_equivalence_flagship.py MODE=trace:
blockify shortens the graph's critical paths, which reorders the heuristic MLP
packing), so the outputs differ at the fp-accumulation floor.  This script
measures that floor directly rather than asserting bit-identity.

Memory-frugal: compiles and runs the two models SEQUENTIALLY (one ~d=8192 model
resident at a time), caching the chain logits to disk between.

GPU + wall-time + host-RAM heavy — run on Modal:
    make modal-run MODULE=scripts.block_forward_divergence \
        MODAL_RUN_GPU_MEMORY=131072 MODAL_RUN_TIMEOUT=7200

Config overrides mirror block_equivalence_flagship.py (D, D_HEAD, D_ROT,
D_HIDDEN, CONFIG, MAX_SEQ_LEN); PREFILL sets the prefill string after <bos>.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _resolve_config(config_name: str) -> str:
    """Find ``<config_name>.yaml`` across local and Modal layouts.

    Local: ``<umbrella>/torchwright_doom/configs/``.  Modal: ``/root/configs/``
    (see modal_image.py add_local_dir)."""
    for cand in (
        Path("/root/configs") / f"{config_name}.yaml",
        _UMBRELLA / "torchwright_doom" / "configs" / f"{config_name}.yaml",
    ):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        f"config {config_name}.yaml not found in /root/configs or "
        f"{_UMBRELLA / 'torchwright_doom' / 'configs'}"
    )


def _resolve_wad(cfg):
    from torchwright_doom.inference.config import resolve_wad_path

    try:
        return resolve_wad_path(cfg, base_dir=str(_UMBRELLA / "torchwright_doom"))
    except Exception:
        for cand in (
            Path("/root/doom1.wad"),
            Path("/root/configs/doom1.wad"),
            _UMBRELLA / "torchwright_doom" / "doom1.wad",
            _UMBRELLA / "doom1.wad",
            Path.home() / "Downloads" / "doom1.wad",
        ):
            if cand.exists():
                return str(cand)
        return None


def _build(config_name, d_head, d_rot, max_seq_len, *, blockify_it):
    """Build the flagship graph + production width-safe fusion; optionally
    blockify.  Returns (output_node, embedding)."""
    from torchwright.graph.optimize import fuse_consecutive_linears
    from torchwright_doom.inference.compiled_model import build_graph
    from torchwright_doom.inference.config import load_render_config

    cfg = load_render_config(_resolve_config(config_name))
    next_token, _rope, emb, _banks = build_graph(
        d_head=d_head,
        max_positions=max_seq_len,
        d_rot=d_rot,
        asset_config=cfg.asset_config(),
        wad_path=_resolve_wad(cfg),
    )
    fuse_consecutive_linears({next_token}, verbose=False, skip_relu_ejecting=True)
    if blockify_it:
        from torchwright.graph.blockify import blockify

        next_token = blockify(next_token, verbose=True)
    return next_token, emb


def _compile_and_run(
    config_name,
    d,
    d_head,
    d_rot,
    d_hidden,
    max_seq_len,
    *,
    blockify_it,
    prefill,
    onnx_path,
):
    """Compile one path to ONNX and run the prefill; return the logits tensor."""
    import torch

    from torchwright.compiler.export import compile_to_onnx
    from torchwright.compiler.onnx_load import load_onnx

    label = "block" if blockify_it else "chain"
    t0 = time.perf_counter()
    out, emb = _build(config_name, d_head, d_rot, max_seq_len, blockify_it=blockify_it)
    print(f"[{label}] built graph in {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    compile_to_onnx(
        out,
        embedding=emb,
        output_path=onnx_path,
        d=d,
        d_head=d_head,
        d_hidden=d_hidden,
        max_seq_len=max_seq_len,
        optimize=0,
        assume_zero_init=True,
        rms_norm_const_exp=63,
        verbose=False,
    )
    print(f"[{label}] compiled to ONNX in {time.perf_counter() - t0:.1f}s", flush=True)

    model = load_onnx(onnx_path)
    tokens = ["<bos>"] + list(prefill)
    ids = torch.tensor([model.token_to_id(t) for t in tokens], dtype=torch.int64)
    t0 = time.perf_counter()
    logits = model(ids)
    print(
        f"[{label}] prefill {len(tokens)} tokens in "
        f"{time.perf_counter() - t0:.1f}s -> logits {tuple(logits.shape)}",
        flush=True,
    )
    return logits.detach().cpu()


def main() -> None:
    import torch

    config_name = os.environ.get("CONFIG", "e1m1")
    d = int(os.environ.get("D", "8192"))
    d_head = int(os.environ.get("D_HEAD", "128"))
    d_rot = int(os.environ["D_ROT"]) if os.environ.get("D_ROT") else 64
    d_hidden = int(os.environ.get("D_HIDDEN", "16384"))
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "65536"))
    prefill = os.environ.get("PREFILL", "e1m1")

    print(
        f"Flagship forward divergence: config={config_name} d={d} d_head={d_head} "
        f"d_rot={d_rot} d_hidden={d_hidden} prefill={prefill!r}",
        flush=True,
    )

    tmp = Path(tempfile.mkdtemp(prefix="block_fwd_"))
    chain_logits = _compile_and_run(
        config_name,
        d,
        d_head,
        d_rot,
        d_hidden,
        max_seq_len,
        blockify_it=False,
        prefill=prefill,
        onnx_path=str(tmp / "chain.onnx"),
    )
    block_logits = _compile_and_run(
        config_name,
        d,
        d_head,
        d_rot,
        d_hidden,
        max_seq_len,
        blockify_it=True,
        prefill=prefill,
        onnx_path=str(tmp / "block.onnx"),
    )

    assert (
        chain_logits.shape == block_logits.shape
    ), f"shape mismatch {chain_logits.shape} vs {block_logits.shape}"
    diff = (chain_logits - block_logits).abs()
    max_div = float(diff.max())
    mean_div = float(diff.mean())
    chain_arg = chain_logits.argmax(dim=-1)
    block_arg = block_logits.argmax(dim=-1)
    argmax_agree = bool((chain_arg == block_arg).all())
    n_pos = chain_logits.shape[0]
    n_arg_disagree = int((chain_arg != block_arg).sum())

    print("\n" + "=" * 60)
    print("Flagship forward divergence (chain-mined vs blockified)")
    print("=" * 60)
    print(f"  logits shape: {tuple(chain_logits.shape)}  (positions x vocab)")
    print(f"  max |logit divergence|: {max_div:.6e}")
    print(f"  mean |logit divergence|: {mean_div:.6e}")
    print(
        f"  per-position argmax agreement: {argmax_agree} "
        f"({n_pos - n_arg_disagree}/{n_pos} positions match)"
    )


if __name__ == "__main__":
    main()
