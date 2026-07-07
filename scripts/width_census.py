"""W0 resident-width floor census — the go/no-go gate for the d=4096 track.

Answers one question: how many residual-stream columns can NEVER be
reclaimed by the scheduler, and who owns them?  If the permanent residents
plus the widest unavoidable transient working set already crowd 4,096
columns, no amount of transient shrinkage (radix floors, two-level ray
count) makes d=4096 schedulable, and the width plan stops.

Method: run the heuristic scheduler in schedule-only mode exactly like
``analyze_forward_cost.schedule_only_capture`` (production fusion pre-pass,
no weight tensors), but capture the warm-start's residual map INSTANCE so
we can read its state after the schedule completes.  Three reports:

  1. **Permanent residents** — nodes still holding columns when the whole
     forward pass has been scheduled.  Split into pre-schedule seeds (the
     input/embedding row + the never-freed RoPE const-one column) and
     nodes born during the schedule that no layer ever freed.
  2. **Reserved columns** — withheld from the free pool for the whole
     compile (the pinned-constant RMSNorm columns).
  3. **The peak live-set, split permanent vs transient** — of the columns
     live at the width peak, how many belong to permanent residents
     (unremovable by transient-shrinking work) vs transients.

Production topology needs the screen env vars (the screen-env trap —
see swiglu_opportunities_findings.md, Method).  This script *defaults*
them to the production 320x200 low-detail HUD-on values so a bare run
measures the real graph; override via the environment as usual.

Usage (CPU, ~2 min):
    uv run python -m scripts.width_census                  # d=8192 production
    D=8192 D_HEAD=128 DH=16384 uv run python -m scripts.width_census
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
# Default to the PRODUCTION graph topology (the screen-env trap: without
# these you silently measure a 60x50 HUD-off graph).
os.environ.setdefault("TORCHWRIGHT_DOOM_RENDER_SCALE", "1")
os.environ.setdefault("TORCHWRIGHT_DOOM_SCREEN_WIDTH", "320")
os.environ.setdefault("TORCHWRIGHT_DOOM_SCREEN_HEIGHT", "200")
os.environ.setdefault("TORCHWRIGHT_DOOM_DETAIL", "low")
os.environ.setdefault("TORCHWRIGHT_DOOM_HUD", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

import torchwright.compiler.forward.compile as _fc
from torchwright.graph.misc import LiteralValue
from torchwright.graph.node import reserve_node_id_above
from torchwright.ops.inout_nodes import create_rope_config
from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward

from scripts.analyze_forward_cost import _apply_fusion, _node_label, bucket


def main():
    d = int(os.environ.get("D", "8192"))
    d_head = int(os.environ.get("D_HEAD", "128"))
    d_hidden = int(os.environ.get("DH", "16384"))
    run_og = os.environ.get("OPT_GRAPH", "1") == "1"
    admit = float(os.environ.get("ADMIT", "0.4"))
    # RESERVE=n withholds n columns before scheduling, mirroring the
    # production compile's RMSNorm pinned-constant reservation (1 column at
    # even log2(d), 2 at odd) — the probe-vs-production feasibility gap at
    # razor-thin widths.
    reserve = int(os.environ.get("RESERVE", "0"))

    print(
        f"[env] screen={os.environ['TORCHWRIGHT_DOOM_SCREEN_WIDTH']}x"
        f"{os.environ['TORCHWRIGHT_DOOM_SCREEN_HEIGHT']} "
        f"scale={os.environ['TORCHWRIGHT_DOOM_RENDER_SCALE']} "
        f"detail={os.environ['TORCHWRIGHT_DOOM_DETAIL']} "
        f"hud={os.environ['TORCHWRIGHT_DOOM_HUD']} | "
        f"d={d} d_head={d_head} d_hidden={d_hidden} optimize_graph={run_og}"
    )

    emb = build_doom_embedding("token_ids")
    rope = create_rope_config(d_head=d_head, max_positions=65536, d_rot=d_head // 2)
    nt = forward(emb, GraphPast(input_vec=emb, rope=rope))

    n_fused = None
    if run_og:
        n_fused = _apply_fusion(nt, verbose=False, eject_budget=d - 1)

    from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
    from torchwright.compiler.forward.residual_map import ResidualStreamMap
    from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
    from torchwright.compiler.lower import lower

    graph = GraphAnalyzer(lower(nt).output_node)
    output_node = graph.get_output_node()
    input_nodes = [n for n in graph.get_all_nodes() if graph.is_input_node(n)]
    rmap = ResidualStreamMap(d)
    if reserve:
        rmap.reserve(range(d - reserve, d))
    # Mirror forward_compile's RoPE-era residual seed (see
    # analyze_forward_cost.schedule_only_capture for the rationale).
    reserve_node_id_above(graph.get_all_nodes())
    const_one = LiteralValue(torch.ones(1), name="rope_self_match_const_one")
    rmap.allocate(const_one)
    rmap.mark_clean(rmap.get_indices(const_one))
    for n in input_nodes:
        rmap.allocate(n)
        rmap.mark_clean(rmap.get_indices(n))
    computed = set(input_nodes)
    policy = SchedulingPolicy()
    seed_ids = {const_one.node_id} | {n.node_id for n in input_nodes}

    # Capture the warm-start's cloned residual map: _run_heuristic_warm_start
    # deep-copies our rmap into a _TrackingResidualStreamMap and mutates only
    # the clone, so the end-of-schedule live set exists only on that instance.
    captured: dict = {}
    peak = {"used": 0, "live": [], "layer": -1}
    _OrigTracking = _fc._TrackingResidualStreamMap

    class _CapturingMap(_OrigTracking):  # type: ignore[misc, valid-type]
        def __init__(self, base):
            super().__init__(base)
            captured["map"] = self

        def allocate(self, node):
            res = super().allocate(node)
            used = self.d - len(self._free)
            if used > peak["used"]:
                peak["used"] = used
                peak["layer"] = self.current_layer
                peak["live"] = list(self._node_to_indices)
            return res

    _fc._TrackingResidualStreamMap = _CapturingMap
    try:
        t0 = time.perf_counter()
        layers, _routing, cancel, n_layers = _fc._run_heuristic_warm_start(
            graph=graph,
            d=d,
            d_head=d_head,
            pos_encoding=None,
            d_hidden=d_hidden,
            residual_map=rmap,
            computed=computed,
            clusters=None,
            admission_budget_fraction=admit,
            policy=policy,
            output_node=output_node,
            max_layers=600,
        )
        t = time.perf_counter() - t0
    finally:
        _fc._TrackingResidualStreamMap = _OrigTracking
    if n_layers == 0:
        # Deadlock diagnosis: the warm start caught the scheduler's
        # no-progress RuntimeError, so the captured map holds the exact
        # allocator state at the blocked layer.
        fmap = captured["map"]
        live = sorted(fmap._node_to_indices.items(), key=lambda kv: -len(kv[1]))
        used = fmap.d - len(fmap._free)
        print(
            f"\n=== DEADLOCK at d={d}, layer {fmap.current_layer}: "
            f"{used} cols live, {len(fmap._free)} free ==="
        )
        grouped: dict[str, list[int]] = defaultdict(list)
        for node, cols in live:
            grouped[_node_label(node) or "?"].append(len(cols))
        rows = sorted(
            ((nm, len(ws), sum(ws)) for nm, ws in grouped.items()),
            key=lambda r: -r[2],
        )
        print("    live set at deadlock (count x width = total):")
        for nm, cnt, tot in rows[:30]:
            print(f"  {tot:6d} = {cnt:4d} x {tot // cnt:<5d}  {nm}")
        raise RuntimeError(f"heuristic deadlocked at d={d} (schedule-only)")

    fmap = captured["map"]
    print(
        f"\n=== schedule-only: layers={n_layers} fused={n_fused} "
        f"peak_width={peak['used']}@L{peak['layer']} t={t:.1f}s ==="
    )

    # ---- 1. permanent residents: still allocated after the last layer ----
    def _birth(node) -> str:
        if node.node_id in seed_ids:
            return "seed"
        li = layers.get(node.node_id)
        return f"L{li}" if li is not None else "?"

    final_live = sorted(fmap._node_to_indices.items(), key=lambda kv: -len(kv[1]))
    perm_cols = sum(len(cols) for _, cols in final_live)
    seed_cols = sum(len(cols) for node, cols in final_live if node.node_id in seed_ids)
    out_cols = sum(
        len(cols) for node, cols in final_live if node.node_id == output_node.node_id
    )
    print(
        f"\n--- permanent residents (never freed): {perm_cols} cols "
        f"across {len(final_live)} nodes ---"
    )
    print(
        f"    seeds (input row + const-one): {seed_cols} | "
        f"output node: {out_cols} | "
        f"other never-freed: {perm_cols - seed_cols - out_cols} | "
        f"reserved (rms pinned): {len(fmap._reserved)}"
    )
    for node, cols in final_live[:40]:
        tag = (
            "OUTPUT"
            if node.node_id == output_node.node_id
            else ("seed" if node.node_id in seed_ids else "held")
        )
        print(
            f"  {len(cols):6d}  born {_birth(node):>5}  [{tag:6s}] "
            f"{bucket(_node_label(node)):18s} {_node_label(node)}"
        )
    if len(final_live) > 40:
        rest = sum(len(c) for _, c in final_live[40:])
        print(f"  ... {len(final_live) - 40} more nodes, {rest} cols")

    # ---- 1b. the seed rows: how long does the input row stay resident? ----
    # A seed (the input/embedding row) that is freed late is *effectively*
    # permanent: its columns coexist with everything scheduled before its
    # last consumer.  ``cancel`` maps node_id -> the layer that freed it.
    print("\n--- seed rows (input/embedding + const-one): width and free layer ---")
    for n in [const_one, *input_nodes]:
        freed = cancel.get(n.node_id)
        span = "never freed" if freed is None else f"freed L{freed}"
        print(f"  {len(n):6d}  {span:>12}  {_node_label(n)}")

    # ---- 2. the peak live-set, split permanent vs transient ----
    perm_nodes = set(fmap._node_to_indices)
    peak_perm = sum(len(n) for n in peak["live"] if n in perm_nodes)
    peak_trans = sum(len(n) for n in peak["live"] if n not in perm_nodes)
    print(
        f"\n--- peak live-set (L{peak['layer']}, {peak['used']} cols): "
        f"{peak_perm} permanent + {peak_trans} transient ---"
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for n in peak["live"]:
        if n not in perm_nodes:
            grouped[_node_label(n) or "?"].append(len(n))
    rows = sorted(
        ((nm, len(ws), sum(ws)) for nm, ws in grouped.items()), key=lambda r: -r[2]
    )
    print("    top transients at the peak (count x width = total):")
    for nm, cnt, tot in rows[:25]:
        print(f"  {tot:6d} = {cnt:4d} x {tot // cnt:<5d}  {nm}")

    # ---- 3. the go/no-go arithmetic ----
    # The three residency windows are disjoint in time; none of them adds to
    # the mid-schedule peak except the whole-pass columns.  The go/no-go
    # question is whether any single window crowds 4,096 on its own.
    whole_pass = sum(
        len(cols) for node, cols in final_live if node.node_id in seed_ids
    ) + len(fmap._reserved)
    input_row = sum(len(n) for n in input_nodes)
    input_freed = max((cancel.get(n.node_id, n_layers) for n in input_nodes), default=0)
    late_births = sorted(
        (layers.get(node.node_id, -1) for node, _ in final_live), reverse=True
    )
    end_window = perm_cols - (whole_pass - len(fmap._reserved))
    print(
        f"\n--- W0 verdict input (disjoint residency windows) ---\n"
        f"  whole-pass:  {whole_pass:5d} cols (never-freed seeds + reserved)\n"
        f"  input row:   {input_row:5d} cols, live L0..L{input_freed}\n"
        f"  end-of-pass: {end_window:5d} cols (output row; births "
        f"{[f'L{b}' for b in late_births[:4]]})\n"
        f"  widest window = {max(whole_pass, input_row, end_window)} cols "
        f"({max(whole_pass, input_row, end_window) / 4096:.1%} of d=4096)"
    )


if __name__ == "__main__":
    main()
