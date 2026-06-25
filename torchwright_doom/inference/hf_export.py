"""Export the compiled DOOM ONNX artifact as a native HuggingFace
``TorchwrightForCausalLM`` bundle, and prove token-identical parity against the
``OnnxTokenModule`` oracle.

This is the DOOM-specific consumer of torchwright's generic HF core
(``torchwright.compiler.hf``): the converter, config, and model are reused
unchanged; this module only adds the DOOM surface — the ``DoomTokenizer`` half
of the bundle, the screen-config / vocab-fingerprint stamp, and a parity check
driven from the DOOM prefill fixture.

It runs where the artifact and a big GPU live (Modal — see
``modal_render.py::hf_export``): the densified fp32 weights are ~28 GB and an
unbounded KV cache over a full rollout is tens of GB, neither of which fits a
local card.

**Screen-config caveat.** The DOOM token vocab is built at import time from the
screen-size env vars (``embedding``/``vocab``). The artifact was compiled at one
screen config, so :func:`apply_screen_env` MUST run before any of those modules
import. Every DOOM import in this file is therefore deferred into the functions,
which the caller invokes only after applying the screen env.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch


def _meta_path(onnx_path: str | Path) -> Path:
    # Canonical sidecar naming (``<stem>.meta.json`` — the .onnx suffix is
    # replaced, not appended); reuse the exporter's own helper.
    from torchwright.compiler.export import meta_path_for

    return Path(meta_path_for(str(onnx_path)))


def _doom_bos_eos_strings(meta: dict) -> tuple[str, str]:
    """The artifact-vocab strings for the BEGIN / DONE rows.

    DOOM's bos/eos are the ``begin`` / ``done`` token *types*, but the converter
    looks the bos/eos up by their string in the artifact's ``meta["vocab"]`` —
    whose labels (``value(v=-1)`` ...) are the compiler's, not the tokenizer's.
    So resolve the rows from the token types and read the compiler's label at
    those rows, guarding that each label is unique (the converter does
    ``vocab.index(...)``, which would pick the wrong row on a collision).
    """
    from ..inference.tokens_bridge import token_to_row
    from ..tokens import Token
    from ..vocab import BEGIN, DONE

    vocab: list[str] = list(meta["vocab"])
    begin_row = token_to_row(Token(BEGIN, {}))
    done_row = token_to_row(Token(DONE, {}))
    bos_str, eos_str = vocab[begin_row], vocab[done_row]
    assert vocab.count(bos_str) == 1, f"bos label {bos_str!r} not unique in vocab"
    assert vocab.count(eos_str) == 1, f"eos label {eos_str!r} not unique in vocab"
    # The converter must map these strings back to exactly these rows.
    assert vocab.index(bos_str) == begin_row
    assert vocab.index(eos_str) == done_row
    return bos_str, eos_str


def export_bundle(onnx_path: str | Path, save_dir: str | Path, config) -> dict[str, Any]:
    """Convert the DOOM artifact and write a full trust-remote-code bundle.

    Writes the native model (safetensors + ``config.json`` + the shipped
    torchwright modeling/config files) and the ``DoomTokenizer`` bundle into
    ``save_dir``, then stamps the screen config + vocab fingerprint into
    ``config.json`` so ``from_pretrained`` can fail loud on a screen mismatch.

    Returns a small summary dict (no weights).
    """
    from torchwright.compiler.hf.convert import save_bundle

    from ..embedding import TOKEN_VOCAB
    from ..tokenizer.hf_tokenizer import (
        DoomTokenizer,
        screen_config,
        vocab_fingerprint,
    )

    onnx_path = str(onnx_path)
    save_dir = str(save_dir)
    meta = json.loads(_meta_path(onnx_path).read_text())
    bos_str, eos_str = _doom_bos_eos_strings(meta)

    # The generic tokenizer is wrong for DOOM; write the model only here and the
    # DoomTokenizer separately below.
    model = save_bundle(
        onnx_path,
        save_dir,
        bos_token=bos_str,
        eos_token=eos_str,
        write_tokenizer=False,
    )

    # Sanity: the model's logit width must equal the DOOM vocab row count (no
    # over-allocation here — unlike the calculator, every DOOM row is a token).
    assert model.config.vocab_size == TOKEN_VOCAB.n_rows, (
        f"model vocab_size {model.config.vocab_size} != TOKEN_VOCAB.n_rows "
        f"{TOKEN_VOCAB.n_rows}; model and tokenizer disagree on the vocab"
    )

    tok = DoomTokenizer(asset_config=config.asset_config())
    tok.save_pretrained(save_dir)

    # Stamp the screen identity so a reload under a different screen config (which
    # would silently re-key the vocab) fails loud instead.
    cfg_path = Path(save_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["doom_screen_config"] = screen_config()
    cfg["doom_vocab_fingerprint"] = vocab_fingerprint()
    cfg_path.write_text(json.dumps(cfg, indent=2))

    return {
        "save_dir": save_dir,
        "vocab_size": int(model.config.vocab_size),
        "n_layers": int(model.config.n_layers),
        "bos_token_id": int(model.config.bos_token_id),
        "eos_token_id": int(model.config.eos_token_id),
        "screen_config": screen_config(),
        "vocab_fingerprint": vocab_fingerprint(),
    }


def _greedy_hf(
    model,
    prefill_ids: list[int],
    max_positions: int,
    terminal_row: int,
    device,
    progress_every: int = 0,
) -> list[int]:
    """Greedy argmax rollout from the native HF model (stock unbounded
    ``DynamicCache``), mirroring :meth:`TokenRuntime.pure_ar_rollout` exactly:
    seed from the prefill's last logit, then decode one row at a time until the
    terminal row or ``max_positions`` emitted rows.
    """
    model.eval()
    ids = torch.tensor(prefill_ids, dtype=torch.long, device=device)[None, :]
    t0 = time.time()
    with torch.no_grad():
        res = model(input_ids=ids, use_cache=True)
        past = res.past_key_values
        cur = int(res.logits[0, -1].argmax())
        emitted = [cur]
        while cur != terminal_row and len(emitted) < max_positions:
            nxt = torch.tensor([[cur]], dtype=torch.long, device=device)
            res = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = res.past_key_values
            cur = int(res.logits[0, -1].argmax())
            emitted.append(cur)
            if progress_every and len(emitted) % progress_every == 0:
                dt = time.time() - t0
                print(
                    f"[hf_render] {len(emitted)} tokens ({dt:.0f}s, "
                    f"{dt / len(emitted) * 1000:.0f} ms/tok)",
                    flush=True,
                )
    return emitted


def check_parity(
    cache_dir: str | Path,
    prefill_ids: list[int],
    max_positions: int = 256,
    device: str = "cpu",
    expiring_types: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Token-identical parity: the native HF greedy rollout vs the production
    ONNX runtime (``OnnxTokenRuntime.pure_ar_rollout``) from a given DOOM prefill
    (e.g. the ``TINY_BSP_SCENE`` fixture rows), capped at ``max_positions`` rows.

    The oracle is the *real production windowed runtime* — the static-cache
    contract loader can't drive a windowed+expiry artifact. ``max_positions`` is
    kept well under the cache window (12288), so the production runtime evicts
    nothing and keeps every row identity-placed — exactly what the HF unbounded
    ``DynamicCache`` does — making the streams comparable. Returns a report dict;
    raises ``AssertionError`` on the first divergent row (with context).
    """
    from torchwright.compiler.hf.convert import convert_onnx_to_hf

    from ..vocab import DONE
    from .compile_cache import load_cached_runtime
    from .tokens_bridge import row_index

    cache_dir = Path(cache_dir)
    onnx_path = str(cache_dir / "model.onnx")
    meta = json.loads(_meta_path(onnx_path).read_text())
    bos_str, eos_str = _doom_bos_eos_strings(meta)
    terminal_row = row_index(DONE, {})

    # Oracle: production windowed runtime, greedy AR.
    oracle = load_cached_runtime(cache_dir, expiring_types=expiring_types)
    oracle_res = oracle.pure_ar_rollout(
        list(prefill_ids), max_positions, terminal_row
    )
    oracle_rows = oracle_res.emitted_rows

    # Native HF: convert (fp32) + greedy, mirroring pure_ar_rollout semantics.
    model = convert_onnx_to_hf(onnx_path, bos_token=bos_str, eos_token=eos_str)
    model = model.to(torch.float32).eval()
    if device != "cpu":
        model = model.to(device)
    hf_rows = _greedy_hf(model, list(prefill_ids), max_positions, terminal_row, device)

    first_div = None
    for i, (a, b) in enumerate(zip(oracle_rows, hf_rows)):
        if a != b:
            first_div = i
            break
    if first_div is None and len(oracle_rows) != len(hf_rows):
        first_div = min(len(oracle_rows), len(hf_rows))

    report = {
        "max_positions": max_positions,
        "prefill_len": len(prefill_ids),
        "terminal_row": terminal_row,
        "n_oracle_rows": len(oracle_rows),
        "n_hf_rows": len(hf_rows),
        "oracle_stopped": oracle_res.stopped,
        "token_identical": first_div is None,
        "first_divergence": first_div,
        "device": device,
    }
    assert first_div is None, (
        f"parity FAILED at row {first_div}: "
        f"oracle={oracle_rows[first_div] if first_div < len(oracle_rows) else None} "
        f"hf={hf_rows[first_div] if first_div < len(hf_rows) else None} "
        f"(lens oracle={len(oracle_rows)} hf={len(hf_rows)})"
    )
    return report


def render_frame(
    cache_dir: str | Path,
    config,
    out_dir: str | Path,
    *,
    device: str = "cpu",
    max_positions: int,
    base_dir: str | Path | None = None,
    x: float | None = None,
    y: float | None = None,
    angle: int | None = None,
    viewz: float | None = None,
    png_zoom: int = 8,
    progress_every: int = 500,
) -> dict[str, Any]:
    """Render one full DOOM frame with the native HF model and compare it to the
    pydoom reference renderer — the same ground-truth gate the production runtime
    passes.

    Converts the artifact to ``TorchwrightForCausalLM`` (fp32), runs the greedy
    rollout from the pose's prefill to the terminal row (or ``max_positions``),
    decodes the row stream to pixels, and scores it against the pydoom reference
    (coverage + within-option color), writing ``generated/reference/diff.png``.
    Returns a small report dict (no weights, no pixels).
    """
    from torchwright.compiler.hf.convert import convert_onnx_to_hf

    from ..vocab import DONE
    from . import compare as compare_mod
    from .decode import decode_rows_to_pixels
    from .tokens_bridge import row_index
    from .wad_scene import (
        load_render_scene,
        pose_from_world,
        prefill_rows_for,
        pydoom_scene_for,
    )

    cache_dir = Path(cache_dir)
    onnx_path = str(cache_dir / "model.onnx")
    meta = json.loads(_meta_path(onnx_path).read_text())
    bos_str, eos_str = _doom_bos_eos_strings(meta)
    terminal_row = row_index(DONE, {})

    scene = load_render_scene(config, base_dir=base_dir)
    pose = pose_from_world(scene, x=x, y=y, angle=angle, viewz=viewz)
    prefill_ids = prefill_rows_for(scene, pose)
    print(
        f"[hf_render] prefill={len(prefill_ids)} rows, max_positions={max_positions}",
        flush=True,
    )

    model = convert_onnx_to_hf(onnx_path, bos_token=bos_str, eos_token=eos_str)
    model = model.to(torch.float32).eval()
    if device != "cpu":
        model = model.to(device)

    t0 = time.time()
    hf_rows = _greedy_hf(
        model, prefill_ids, max_positions, terminal_row, device, progress_every
    )
    seconds = time.time() - t0
    stopped = "terminal" if (hf_rows and hf_rows[-1] == terminal_row) else "cap"
    print(
        f"[hf_render] {len(hf_rows)} tokens in {seconds:.0f}s, stopped={stopped}",
        flush=True,
    )

    gen = decode_rows_to_pixels(hf_rows, palette=scene.asset_book.palette)
    py_scene = pydoom_scene_for(scene, pose)
    py_pose = py_scene.test_poses[0]
    ref = compare_mod.reference_pixels(py_scene, py_pose)
    options = compare_mod.reference_options(py_scene, py_pose)
    report = compare_mod.compare(gen, ref, options)
    pngs = compare_mod.write_pngs(gen, ref, out_dir, options=options, scale=png_zoom)

    return {
        "n_rows": len(hf_rows),
        "stopped": stopped,
        "seconds": seconds,
        "report_text": report.format_short(),
        "pngs": [p.name for p in pngs],
    }
