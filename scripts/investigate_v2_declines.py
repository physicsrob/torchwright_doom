"""Forensic dump for v2 Phase-A declines that sit INSIDE honest domains.

Three declines from the 2026-07-06 doom v2 report cannot be explained by
interval-arithmetic domain slack (their worst points are values the real
render can produce), so each needs its bit-level reason (D2) before any
certificate fix or synthesis work builds on top:

* ``split_5`` / ``table_lookup_2d_col_step`` deviates exactly 2.00 at
  x=934.5 — the W=2-bounded step value, suggesting a whole step
  transition missing from the composed candidate set (pullback gap?).
* ``sub`` / ``clamp_0_2`` deviates 0.249 at x=4089.22 inside a real
  ±8191.94 domain — suspected fp32 position-quantization tilt
  propagating beyond the ±1-segment window the resolution floor checks.
* ``Linear#5848`` / ``table_lookup_2d_row_deltas`` deviates 66.8 at
  x=6.99994 — one fp32 step left of an integer floor boundary, inside
  the ±1023 native contract.

For each target this script re-certifies the subgraph with
``keep_raw=True`` and dumps the neighborhood of the worst point:
candidate kinks, measurement-frame knots/values, simplified knots,
a dense fp32 oracle sweep vs both functions, the true transition
positions the sweep reveals, gate-crossing recomputation for FFN
members (walk frame vs empirical), and the resolution-floor inputs
(segment slopes / eps32) at the worst sample.

GPU-free but heavy (full graph build + lower), so run it on Modal:

    make modal-run MODULE=scripts.investigate_v2_declines CPU_ONLY=1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# (source name, member count, declining member name, worst x from the
# report) — the match keys for the three target subgraphs.  The two
# 38-member Linear-source subgraphs share a decline member name; both
# are dumped (one doubles as the interval-slack confirmation).
TARGETS = [
    ("split_5", 21, "table_lookup_2d_col_step", 934.5),
    ("sub", 10, "clamp_0_2", 4089.22),
    (None, 38, "table_lookup_2d_row_deltas", 6.99994),
    (None, 38, "table_lookup_2d_row_deltas", 853258.0),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e1m1.yaml")
    args = ap.parse_args()

    # Screen env BEFORE any graph-module import (the screen-env trap).
    from torchwright_doom.inference.config import (
        apply_screen_env,
        load_render_config,
    )

    cfg = load_render_config(args.config)
    apply_screen_env(cfg)

    import torch

    from torchwright.compiler.collapse import _seeded_oracle, scalar_sources
    from torchwright.compiler.graph_clone import topological_order
    from torchwright.compiler.lower import lower
    from torchwright.compiler.pl_function import certify_subgraph
    from torchwright.graph.ffn import FFN
    from torchwright_doom.inference.compiled_model import build_graph

    m = cfg.model
    print(f"config {args.config}: d={m.d} d_hidden={m.d_hidden}")
    out, _rope, _emb, _banks = build_graph(
        d_head=m.d_head,
        max_positions=m.max_seq_len,
        d_rot=m.d_rot,
        wad_path=cfg.wad,
    )
    lane_cap = (m.d_hidden if m.d_hidden else m.d) // 4
    lowered = lower(out, collapse_univariate=True, collapse_lane_cap=lane_cap)
    order = topological_order(lowered.output_node)
    src = scalar_sources(order)
    by_src: dict = {}
    for n in order:
        s = src[n]
        if s is not None and s is not n:
            by_src.setdefault(s, []).append(n)
    topo_index = {n: i for i, n in enumerate(order)}

    F64 = torch.float64

    def snap32(t):
        return t.to(torch.float32).to(F64)

    cert_cache: dict = {}

    def certified(source, members):
        key = id(source)
        if key not in cert_cache:
            cert_cache[key] = certify_subgraph(source, members, keep_raw=True)
        return cert_cache[key]

    def dump_target(source, members, member_name, x_star):
        src_name = source.name or f"{type(source).__name__}#{topo_index[source]}"
        print(f"\n{'=' * 72}")
        print(
            f"=== [{src_name}] {len(members)} members, "
            f"target member {member_name}, x* = {x_star}"
        )
        vr = source.value_type.value_range
        print(f"source domain: [{vr.lo:.6g}, {vr.hi:.6g}]")
        cert = certified(source, members)
        if cert.declined is not None:
            print(f"walk declined outright: {cert.declined}")
            return
        # The matching member whose worst point is nearest x*.
        cands = [
            c for node, c in cert.members.items() if (node.name or "") == member_name
        ]
        if not cands:
            print(f"no member named {member_name} in this subgraph")
            return
        c = min(cands, key=lambda c: abs(c.deviation_at - x_star))
        node = c.node
        print(
            f"member {node.name} ({type(node).__name__}, d={node.d_output}), "
            f"inputs: {[(u.name or type(u).__name__) for u in node.inputs]}"
        )
        print(
            f"n_kinks {c.n_kinks}, dev {c.deviation:.4e} at {c.deviation_at:.6g}, "
            f"banded {c.banded_deviation:.4e} at {c.banded_deviation_at:.6g}, "
            f"in-band {c.fillet_deviation:.4e} at {c.fillet_deviation_at:.6g}"
        )
        x0 = (
            float(c.deviation_at)
            if abs(c.deviation_at - x_star) < 1e-3 * max(1.0, abs(x_star)) + 1.0
            else float(x_star)
        )

        raw = c.fn_raw
        # Window: 12 raw knots either side of x*.
        ki = int(torch.searchsorted(raw.x, torch.tensor(x0, dtype=F64)))
        lo_i, hi_i = max(0, ki - 12), min(raw.n_knots - 1, ki + 12)
        w_lo, w_hi = float(raw.x[lo_i]), float(raw.x[hi_i])
        print(f"window: [{w_lo:.8g}, {w_hi:.8g}] (raw knots {lo_i}..{hi_i})")

        # Dense fp32 oracle sweep: coarse across the window + fine at x*.
        eps32 = 2.0 ** (
            torch.floor(torch.log2(torch.tensor(max(abs(x0), 1.0), dtype=F64))) - 23
        )
        eps32 = float(eps32)
        coarse = torch.linspace(w_lo, w_hi, 1601, dtype=F64)
        fine = torch.linspace(x0 - 64 * eps32, x0 + 64 * eps32, 513, dtype=F64)
        xs = torch.unique(snap32(torch.cat([coarse, fine])))
        vals = _seeded_oracle(members, source, xs.to(torch.float32).reshape(-1, 1))
        truth = vals[node].to(F64)
        err_raw = (truth - raw.eval(xs)).abs()
        err_simp = (truth - c.fn.eval(xs)).abs()
        worst_i = int(err_simp.amax(dim=1).argmax())
        worst_dim = int(err_simp[worst_i].argmax())
        print(
            f"sweep ({xs.numel()} pts): worst |truth - fn| = "
            f"{float(err_simp[worst_i, worst_dim]):.4e} at x={float(xs[worst_i]):.8g} "
            f"dim {worst_dim}; worst vs raw frame = "
            f"{float(err_raw.max()):.4e}"
        )

        # True transitions of the worst dim on the sweep grid.
        tv = truth[:, worst_dim]
        d_tv = (tv[1:] - tv[:-1]).abs()
        big = torch.nonzero(d_tv > max(1e-3, 0.25 * float(d_tv.max()))).reshape(-1)
        kr = c.kinks_raw
        print(f"true transitions (dim {worst_dim}) on sweep grid, vs candidates:")
        shown = 0
        for i in big.tolist():
            xa, xb = float(xs[i]), float(xs[i + 1])
            near = (
                float((kr - (xa + xb) / 2).abs().min()) if kr.numel() else float("inf")
            )
            print(
                f"  [{xa:.8g}, {xb:.8g}] jump {float(tv[i + 1] - tv[i]):+.4f} "
                f"nearest candidate {near:.3e} away"
            )
            shown += 1
            if shown >= 24:
                print(f"  ... {len(big) - shown} more")
                break

        # Candidates + knots in the window.
        in_w = kr[(kr >= w_lo) & (kr <= w_hi)]
        print(f"candidate kinks in window: {in_w.numel()}")
        for v in in_w.tolist()[:30]:
            print(f"  cand {v:.8g}")
        print(f"raw knots {lo_i}..{hi_i} (value = dim {worst_dim}):")
        for i in range(lo_i, hi_i + 1):
            print(
                f"  raw  x={float(raw.x[i]):.10g}  y={float(raw.y[i, worst_dim]):+.6f}"
            )
        sx = c.fn.x
        s_in = torch.nonzero((sx >= w_lo) & (sx <= w_hi)).reshape(-1)
        print(f"simplified knots in window: {s_in.numel()}")
        for i in s_in.tolist()[:30]:
            print(f"  simp x={float(sx[i]):.10g}  y={float(c.fn.y[i, worst_dim]):+.6f}")

        # Resolution-floor inputs at the worst sample (the tilt question):
        # segment slopes around the worst sample's segment, in the RAW frame.
        seg = (
            int(torch.searchsorted(raw.x, torch.tensor(x0, dtype=F64), right=True)) - 1
        )
        seg = max(0, min(raw.n_knots - 2, seg))
        slopes = raw.segment_slopes().abs().amax(dim=1)
        print(
            f"worst-sample segment {seg}: width "
            f"{float(raw.x[seg + 1] - raw.x[seg]):.4e}, eps32(x*) {eps32:.4e}"
        )
        for k in range(max(0, seg - 4), min(int(slopes.shape[0]), seg + 5)):
            print(
                f"  seg {k}: [{float(raw.x[k]):.8g}, {float(raw.x[k + 1]):.8g}] "
                f"w={float(raw.x[k + 1] - raw.x[k]):.3e} "
                f"|slope|={float(slopes[k]):.4e} "
                f"floor(8*s*eps)={8.0 * float(slopes[k]) * eps32:.4e}"
            )

        # Gate-crossing recomputation for FFN members: the walk's frame
        # (input's certified PL) vs the empirical frame (input's oracle
        # values on the sweep grid).
        if isinstance(node, FFN):
            u = node.inputs[0]
            if u is source:
                u_vals = xs.reshape(-1, 1)
                u_fn = None
                print("FFN input is the source (identity)")
            elif u in cert.members:
                u_vals = vals[u].to(F64)
                u_fn = cert.members[u].fn
                print(f"FFN input: member {u.name or type(u).__name__}")
            else:
                u_vals, u_fn = None, None
                print("FFN input is a constant subexpression — skipping phi check")
            if u_vals is not None:
                g = node.gate_proj.t().to(F64)
                b = node.gate_bias.to(F64)
                phi_emp = u_vals @ g + b
                emp = []
                s0, s1 = phi_emp[:-1], phi_emp[1:]
                strad = (s0 * s1) < 0
                for i, lane in torch.nonzero(strad).tolist():
                    if w_lo <= float(xs[i]) <= w_hi:
                        emp.append((float(xs[i]), float(xs[i + 1]), lane))
                print(f"empirical gate crossings in window: {len(emp)}")
                for xa, xb, lane in emp[:20]:
                    near = (
                        float((kr - (xa + xb) / 2).abs().min())
                        if kr.numel()
                        else float("inf")
                    )
                    print(
                        f"  lane {lane}: [{xa:.8g}, {xb:.8g}] "
                        f"nearest candidate {near:.3e} away"
                    )
                if u_fn is not None:
                    phi_walk = u_fn.map_affine(g, b)
                    cr = phi_walk.zero_crossings()
                    cr = cr[(cr >= w_lo) & (cr <= w_hi)]
                    print(f"walk-frame gate crossings in window: {cr.numel()}")
                    for v in cr.tolist()[:20]:
                        print(f"  walk cross {v:.8g}")

    done = set()
    for src_name_want, n_members, member_name, x_star in TARGETS:
        # All structurally matching subgraphs, certified (cached), then
        # the one whose target-member worst point sits nearest x* — the
        # 38-member texel pair share a decline member name and are told
        # apart only by where they deviate.
        matches = []
        for source in sorted(by_src, key=topo_index.__getitem__):
            members = by_src[source]
            if len(members) != n_members:
                continue
            if src_name_want is not None and (source.name or "") != src_name_want:
                continue
            if not any((mm.name or "") == member_name for mm in members):
                continue
            if id(source) in done:
                continue
            cert = certified(source, members)
            if cert.declined is not None:
                continue
            dists = [
                abs(c.deviation_at - x_star)
                for node, c in cert.members.items()
                if (node.name or "") == member_name
            ]
            if dists:
                matches.append((min(dists), source, members))
        if not matches:
            print(f"\n(no unvisited subgraph matched {src_name_want}/{member_name})")
            continue
        _, source, members = min(matches, key=lambda t: t[0])
        done.add(id(source))
        dump_target(source, members, member_name, x_star)


if __name__ == "__main__":
    main()
