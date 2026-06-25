"""Export the compiled DOOM ONNX artifact as a native HuggingFace
``TorchwrightForCausalLM`` bundle (the Hub publish path).

This is the DOOM-specific consumer of torchwright's generic HF core
(``torchwright.compiler.hf``): the converter, config, and model are reused
unchanged; this module only adds the DOOM surface — the ``DoomTokenizer`` half
of the bundle and the screen-config / vocab-fingerprint stamp
(:func:`export_bundle`). The production render + correctness gate live in
``cli.run_config`` / ``hf_runtime`` (``make run`` / ``make run COMPARE=1``).

It runs where the artifact and a big GPU live (Modal — see
``modal_render.py::hf_export``): the densified fp32 weights are ~28 GB, which
does not fit a local card.

**Screen-config caveat.** The DOOM token vocab is built at import time from the
screen-size env vars (``embedding``/``vocab``). The artifact was compiled at one
screen config, so :func:`apply_screen_env` MUST run before any of those modules
import. Every DOOM import in this file is therefore deferred into the functions,
which the caller invokes only after applying the screen env.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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


def export_bundle(
    onnx_path: str | Path, save_dir: str | Path, config
) -> dict[str, Any]:
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
