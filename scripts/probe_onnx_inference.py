"""De-risk: can onnxruntime run the doom token-I/O ONNX within VM memory, and
does its prefill argmax match the exact-math oracle? (Decides whether compile
validation can include inference or only the compile.)"""

from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (
    _UMBRELLA,
    _UMBRELLA / "torchwright_doom",
    _UMBRELLA / "torchwright_doom" / "tests",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import onnxruntime
from torchwright_doom.embedding import TOKEN_VOCAB
from prefill_fixture import TINY_BSP_SCENE, row_index
from torchwright_doom.vocab import SET_CURSOR_DIRECTION_Y

ONNX = os.environ.get("ONNX", "/tmp/doom_forward.onnx")
ids = np.array([row_index(t, s) for t, s in TINY_BSP_SCENE], dtype=np.int64)

print(f"loading {ONNX} ...")
sess = onnxruntime.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
inputs = {i.name: i for i in sess.get_inputs()}
n_layers = sum(1 for n in inputs if n.startswith("past_K_"))
nh = [int(inputs[f"past_K_{i}"].shape[0]) for i in range(n_layers)]
dh = int(inputs["past_K_0"].shape[2])
print(f"session loaded: {n_layers} layers, d_head={dh}, vocab in logits")

feeds = {"token_ids": ids, "past_len": np.array(0, dtype=np.int64)}
for i in range(n_layers):
    feeds[f"past_K_{i}"] = np.zeros((nh[i], 0, dh), dtype=np.float32)
    feeds[f"past_V_{i}"] = np.zeros((nh[i], 0, dh), dtype=np.float32)

logits = sess.run(["logits"], feeds)[0]
print(f"prefill logits {logits.shape}, finite={bool(np.isfinite(logits).all())}")

begin = len(ids) - 1
emitted = int(logits[begin].argmax())
expect = row_index(SET_CURSOR_DIRECTION_Y, {})
print(
    f"BEGIN emits {TOKEN_VOCAB.row_to_token[emitted][0].name} (row {emitted}); "
    f"expect setCursorDirectionY (row {expect}) -> {'OK' if emitted == expect else 'MISMATCH'}"
)
