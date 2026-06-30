"""Plan: BOS at position 0 — the ``debug=True`` forward verification.

Runs ``debug=True`` forwards over the prefill fixture
(``tests.prefill_fixture.TINY_BSP_SCENE``, which now starts with BOS) to confirm,
for a sequence with BOS at position 0:

1. **self-consistency** — the compiled residual stream is internally consistent
   (the +1 global shift introduced no residual-column aliasing). This is the
   compiler-correctness check; a failure here is a real compiler bug (D1).
2. **dispatch** — BOS (position 0) dispatches to the ``no_op`` branch (predicts
   ``NO_OP``) and the prefill->AR seed at BEGIN is unaffected (still
   ``setCursorDirectionY``).
3. **no NEW assert failures from BOS** — the DOOM graph's AR-phase asserts (e.g.
   ``R_StoreWallRange``'s value-range postcondition) run on *every* position
   under ``debug=True``, including prefill positions where those AR nodes compute
   undefined/garbage values. So some graph asserts fire on a prefill-only debug
   forward *independently of BOS*. To isolate BOS's effect we run the control
   (the same prefill with BOS stripped) and require the two to fire the *same*
   assert (same node) — i.e. BOS adds no new failure.

Compiles the full ~85-layer forward (~12 GB), so run it on a GPU:

    make modal-run MODULE=scripts.verify_bos_debug
"""

from __future__ import annotations

import torch

from torchwright.compiler.export import compile_headless

from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.inference.compiled_model import build_graph
from torchwright_doom.inference.tokens_bridge import row_index, rows_to_input
from torchwright_doom.vocab import BEGIN, BOS, NO_OP

from tests.prefill_fixture import TINY_BSP_SCENE

# The same working point the in-process rollout gate compiles at
# (tests/scene/test_forward_ar_rollout.py): the full J forward at d=4096,
# d_head=32, which fits in ~12 GB.
_D = 4096
_D_HEAD = 64  # RoPE: d_head must cover the widest content (~28) on the NoPE tail
_D_ROT = 32
_MAX_POS = 65536


def _pred_type(out_row: torch.Tensor) -> str:
    """Unembed-argmax one compiled output row to its predicted token-type name."""
    wt = W_EMBED.t().to(out_row.device, out_row.dtype)
    row = int((out_row.detach() @ wt).argmax(dim=-1).item())
    return TOKEN_VOCAB.row_to_token[row][0].name


def _debug_forward(compiled, ids: list[int], label: str) -> tuple[str, str]:
    """Run a debug=True forward; return (outcome, detail).

    outcome is "pass" (no asserts), "self_consistency" (compiler bug — D1), or
    "assert:<node>" (a graph assert fired; <node> is the failing node).
    """
    try:
        compiled.step(rows_to_input(ids), compiled.empty_past(), past_len=0, debug=True)
        print(f"[{label}] debug=True PASSED — no asserts fired")
        return ("pass", "")
    except RuntimeError as exc:  # self-consistency failure
        print(f"[{label}] SELF-CONSISTENCY FAILED: {exc}")
        return ("self_consistency", str(exc))
    except AssertionError as exc:
        head = str(exc).split(":")[0].replace("Assert failed at", "").strip()
        print(f"[{label}] graph assert fired at: {head}")
        return (f"assert:{head}", str(exc))


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prefill = [row_index(t, s) for t, s in TINY_BSP_SCENE]
    assert prefill[0] == row_index(BOS, {}), "fixture must start with BOS"
    assert prefill[-1] == row_index(BEGIN, {}), "fixture must end with BEGIN"
    control = prefill[1:]  # same prefill with BOS stripped (the pre-change state)

    next_token, _rope, _emb, _banks = build_graph(
        d_head=_D_HEAD, max_positions=_MAX_POS, d_rot=_D_ROT
    )
    compiled = compile_headless(
        next_token,
        d=_D,
        d_head=_D_HEAD,
        max_layers=200,
        verbose=False,
        device=device,
    )

    # (1) Plain forward (no debug): confirm dispatch + seed.
    out, _past = compiled.step(
        rows_to_input(prefill), compiled.empty_past(), past_len=0
    )
    bos_pred = _pred_type(out[0])
    begin_pred = _pred_type(out[-1])
    print(f"position 0 (BOS) predicts: {bos_pred!r}  (expect {NO_OP.name!r})")
    print(f"last pos (BEGIN) predicts: {begin_pred!r}  (expect 'setCursorDirectionY')")
    assert bos_pred == NO_OP.name, f"BOS did not dispatch to no_op (got {bos_pred!r})"
    assert (
        begin_pred == "setCursorDirectionY"
    ), f"prefill->AR seed at BEGIN changed (got {begin_pred!r})"

    # (2) debug=True with BOS, then the BOS-stripped control.
    bos_outcome, _ = _debug_forward(compiled, prefill, "with-BOS")
    ctl_outcome, _ = _debug_forward(compiled, control, "control (no BOS)")

    if bos_outcome == "self_consistency":
        print("FAIL: self-consistency broke with BOS (compiler bug — D1).")
        return 1
    if bos_outcome == "pass":
        print("PASS: debug=True clean with BOS (self-consistency + all asserts).")
        return 0
    # A graph assert fired with BOS. Harmless iff the control fires the SAME
    # assert (i.e. it is the AR-node-on-prefill-positions behavior, not BOS).
    if bos_outcome == ctl_outcome:
        print(
            f"PASS: BOS introduces no new failure. Both with-BOS and the "
            f"control fire the same pre-existing graph assert ({bos_outcome[7:]}) "
            f"— an AR-phase node asserted at prefill positions, independent of "
            f"BOS. Self-consistency clean; BOS -> no_op; BEGIN seed intact."
        )
        return 0
    print(
        f"FAIL: BOS changed the assert outcome. with-BOS={bos_outcome!r} "
        f"control={ctl_outcome!r} — investigate (BOS may have shifted a value "
        f"out of range)."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
