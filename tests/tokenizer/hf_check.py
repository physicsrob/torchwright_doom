"""HuggingFace ``DoomTokenizer`` checks, run under the trace's screen config.

    python -m tests.tokenizer.hf_check <trace.json.gz>

Prints a JSON report; exits non-zero on any failure. Covers: vocab size ==
``TOKEN_VOCAB.n_rows``; special tokens map onto existing rows (eos==done,
bos==begin, pad==eos, no unk) with no id shift; ``encode`` matches the surface
id stream and ``encode(decode(ids)) == ids`` over the prompt and full rollout;
``save_vocabulary`` / ``from_pretrained`` round-trip plus a tampered-fingerprint
negative test.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile

from torchwright_doom.inference.tokens_bridge import (
    row_index,
    row_to_token,
    token_to_row,
)
from torchwright_doom.tokenizer import surface
from torchwright_doom.tokenizer.hf_tokenizer import DoomTokenizer
from torchwright_doom.tokens import Token
from torchwright_doom.vocab import BOS, DONE, VOCAB_TYPES

_BY_NAME = {t.name: t for t in VOCAB_TYPES}


def _rows(trace_tokens: list[dict]) -> list[int]:
    return [
        token_to_row(Token(_BY_NAME[t["type"]], dict(t["values"])))
        for t in trace_tokens
    ]


def run(trace_path: str) -> dict:
    with gzip.open(trace_path) as fh:
        case = json.load(fh)["cases"][0]
    prompt_rows = _rows(case["prefill_input_tokens"])
    rollout_rows = _rows(case["rollout_output_tokens"])

    tok = DoomTokenizer()
    checks: dict[str, bool] = {}

    checks["vocab_size"] = tok.vocab_size == len(tok._id_to_label)
    # Special tokens map onto existing rows; none appended (len == vocab_size).
    checks["eos_is_done"] = tok.eos_token_id == row_index(DONE, {})
    checks["bos_is_bos"] = tok.bos_token_id == row_index(BOS, {})
    checks["pad_is_eos"] = tok.pad_token_id == tok.eos_token_id
    checks["no_unk"] = tok.unk_token is None
    checks["no_id_shift"] = len(tok) == tok.vocab_size

    for name, rows in (("prompt", prompt_rows), ("rollout", rollout_rows)):
        text = surface.render([row_to_token(r) for r in rows], **tok._knobs)
        ids = tok.encode(text, add_special_tokens=False)
        checks[f"encode_matches_surface_{name}"] = ids == rows
        decoded = tok.decode(ids, skip_special_tokens=False)
        reids = tok.encode(decoded, add_special_tokens=False)
        checks[f"encode_decode_roundtrip_{name}"] = reids == rows

    # save / reload + tampered fingerprint must fail loud.
    directory = tempfile.mkdtemp()
    (path,) = tok.save_vocabulary(directory)
    DoomTokenizer(vocab_file=path)  # clean reload
    checks["save_reload"] = True
    blob = json.loads(open(path).read())
    blob["fingerprint"] = "tampered"
    open(path, "w").write(json.dumps(blob))
    try:
        DoomTokenizer(vocab_file=path)
        checks["tamper_rejected"] = False
    except ValueError:
        checks["tamper_rejected"] = True

    return {"ok": all(checks.values()), "checks": checks}


def main() -> int:
    report = run(sys.argv[1])
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
