"""Teacher-force a published HF bundle on a reference emitted-row stream.

The drift-free disagreement map: the model always sees the REFERENCE
prefix, so each position's greedy argmax is an independent measurement —
autoregressive compounding is removed entirely.  For every emitted
position where the bundle's argmax disagrees with the reference row, this
reports the reference row, the predicted row, both token types, the
top1-top2 logit margin, and the reference row's logit deficit
(logit[top1] - logit[reference]).  A disagreement with a tiny margin is a
decision-boundary flip (numerical-noise territory); a large deficit is a
structural wrong answer.

The reference stream comes from a prior render's retained
``output.ids.json`` (mounted under ``/artifacts/<run-name>/`` on the GPU
container); the bundle from the production HF volume.  Run on Modal:

    TORCHWRIGHT_DOOM_SCREEN_WIDTH=160 TORCHWRIGHT_DOOM_SCREEN_HEIGHT=100 \\
    TORCHWRIGHT_DOOM_RENDER_SCALE=2 TORCHWRIGHT_DOOM_DETAIL=low \\
    TORCHWRIGHT_DOOM_HUD=1 \\
    MODAL_RUN_GPU=B200 MODAL_RUN_GPU_MEMORY=131072 MODAL_RUN_TIMEOUT=7200 \\
    make modal-run MODULE=scripts.teacher_force_margins \\
        ARGS="--bundle <cache-key> --reference-run <artifact-dir-name>"

The screen env vars MUST match the bundle's config: token-row decoding
(``row_to_token``) reads the screen-derived vocabulary layout at import.
The script cross-checks the bundle's frozen vocabulary size against the
local layout and aborts on mismatch rather than mislabeling rows.

Output stays under Modal's 64 KB stdout cap: a full summary + histogram,
and detail lines for at most the first ``--max-detail`` disagreements.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

_BUNDLE_ROOT = Path(
    os.environ.get(
        "TORCHWRIGHT_DOOM_HF_BUNDLE_ROOT", "/root/.cache/torchwright_doom/hf_phi3"
    )
)
_ARTIFACT_ROOT = Path(os.environ.get("TORCHWRIGHT_DOOM_ARTIFACT_ROOT", "/artifacts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="HF bundle cache-key dir name")
    ap.add_argument(
        "--reference-run",
        required=True,
        help="render artifact dir name holding the reference output.ids.json",
    )
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--max-detail", type=int, default=60)
    args = ap.parse_args()

    from torchwright_doom.tokenizer.rows import row_to_token

    bundle = _BUNDLE_ROOT / args.bundle
    reference_path = _ARTIFACT_ROOT / args.reference_run / "output.ids.json"
    inference = json.loads(reference_path.read_text())
    prompt_ids = [int(r) for r in inference["prompt"]["row_ids"]]
    emitted = [int(r) for r in inference["emitted_row_ids"]]
    stream = prompt_ids + emitted

    vocab_words = json.loads((bundle / "doom_vocab.json").read_text())["words"]
    try:
        row_to_token(len(vocab_words) - 1)
    except Exception:
        raise SystemExit(
            f"local token layout is narrower than the bundle vocabulary "
            f"({len(vocab_words)} words) — screen env does not match the bundle"
        )

    from transformers import AutoModelForCausalLM

    print(
        f"[tf] bundle={args.bundle[:12]} reference={args.reference_run} "
        f"prompt={len(prompt_ids)} emitted={len(emitted)}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        bundle,
        attn_implementation="eager",
        dtype=torch.float32,
        disable_mmap=True,
        device_map="cuda",
    ).eval()
    if model.config.vocab_size != len(vocab_words):
        raise SystemExit("bundle vocab_size differs from its frozen word list")

    ids = torch.tensor([stream], dtype=torch.long, device="cuda")
    disagreements: list[dict] = []
    type_counter: Counter[tuple[str, str]] = Counter()
    margin_hist: Counter[str] = Counter()
    past = None
    # Predictions for stream position i live at logits index i; the first
    # emitted row is predicted at position len(prompt_ids) - 1.
    first_target = len(prompt_ids) - 1
    with torch.inference_mode():
        for start in range(0, ids.shape[1], args.chunk):
            piece = ids[:, start : start + args.chunk]
            out = model(
                input_ids=piece,
                past_key_values=past,
                use_cache=True,
                cache_position=torch.arange(
                    start, start + piece.shape[1], device="cuda"
                ),
            )
            past = out.past_key_values
            logits = out.logits[0].float()
            for j in range(piece.shape[1]):
                pos = start + j
                if pos < first_target or pos + 1 >= len(stream):
                    continue
                ref = stream[pos + 1]
                row = logits[j]
                top2_vals, top2_idx = torch.topk(row, 2)
                pred = int(top2_idx[0])
                if pred == ref:
                    continue
                margin = float(top2_vals[0] - top2_vals[1])
                deficit = float(top2_vals[0] - row[ref])
                emit_index = pos + 1 - len(prompt_ids)
                ref_t, pred_t = row_to_token(ref), row_to_token(pred)
                ref_label = f"{ref_t.type.name}{ref_t.values}"
                pred_label = f"{pred_t.type.name}{pred_t.values}"
                type_counter[(ref_t.type.name, pred_t.type.name)] += 1
                bucket = (
                    "<0.01"
                    if deficit < 0.01
                    else (
                        "<0.1"
                        if deficit < 0.1
                        else "<1" if deficit < 1 else "<10" if deficit < 10 else ">=10"
                    )
                )
                margin_hist[bucket] += 1
                disagreements.append(
                    {
                        "emit_index": emit_index,
                        "ref": ref,
                        "pred": pred,
                        "ref_token": ref_label,
                        "pred_token": pred_label,
                        "margin": margin,
                        "deficit": deficit,
                    }
                )
            del logits, out

    print(f"\n[tf] disagreements: {len(disagreements)}/{len(emitted)} emitted rows")
    print("[tf] reference-row logit deficit histogram (how far from flipping back):")
    for bucket in ("<0.01", "<0.1", "<1", "<10", ">=10"):
        if margin_hist[bucket]:
            print(f"       {bucket:>6}: {margin_hist[bucket]}")
    print("[tf] (reference type -> predicted type) counts:")
    for (rt, pt), n in type_counter.most_common(12):
        print(f"       {rt} -> {pt}: {n}")
    print(f"[tf] first {min(args.max_detail, len(disagreements))} disagreements:")
    for d in disagreements[: args.max_detail]:
        print(
            f"  emit[{d['emit_index']}] ref={d['ref']} pred={d['pred']} "
            f"deficit={d['deficit']:.4g} top-margin={d['margin']:.4g}"
        )
        print(f"      ref : {d['ref_token'][:110]}")
        print(f"      pred: {d['pred_token'][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
