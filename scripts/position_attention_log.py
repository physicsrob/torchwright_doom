"""Position-attention usage logger + new-encoding validation harness.

For every place the DOOM graph uses *position* in attention, log what it is
trying to select and which "distractor" positions a position-encoding scheme
must discriminate against — so a candidate replacement encoding can be replayed
offline against real selections before committing to a swap.

It hooks the one chokepoint every position-using op compiles through
(``torchwright.graph.attn.Attn.compute``), drives a token sequence through the
exact-math oracle (``reference_eval``, CPU, no compile), and writes a JSONL log.
Each record isolates the position-independent *content* score from the
*position* score (they live in disjoint qk columns, so ``full = content +
position`` exactly), so the position term can be swapped and the argmax
re-checked.

Two op families are logged:

  * "recency_match"  (attend_most_recent_matching / pick_most_recent):
        content match in some qk columns + a recency term in the raw-counter
        column. The position term is REPLACEABLE; content is frozen. Records
        carry the candidate set so a new scheme can be re-scored and the argmax
        re-checked via ``validate_recency``.

  * "offset"  (attend_to_offset): the WHOLE score is the sinusoidal trig block
        (no content). A new encoding can't be "replayed" here — it must
        intrinsically resolve current_pos+delta. Records are a coverage list of
        the (n_pos, delta) cases actually exercised, plus the peak margin.

Drive modes (what tokens the heads see):

    # tiny BSP scene, free-run (TRAVERSAL SPINE ONLY — recency heads never see
    # their real keys; most log the no-match pure-recency fallback). Fast.
    python -m scripts.position_attention_log --validate

    # REPRESENTATIVE: teacher-force a real frame including rasterization, capped
    # to `--span` AR tokens past the seed so reference_eval stays tractable.
    # Use the low-res config + a span big enough to reach the flat pass.
    python -m scripts.position_attention_log --frame \
        --config configs/e1m1_lowres.yaml --span 4000 --validate

    # teacher-force an explicit token-id list (your own dump)
    python -m scripts.position_attention_log --tokens ids.json --validate

reference_eval is O(n_pos^2) in time and materialises every node at every
position, so a capped span (a few thousand) is the practical envelope on a
workstation; the full ~42k-token e1m1 frame belongs on a big-memory box.

Output: position_attention_log.jsonl (next to this script).
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def _apply_screen_env_from_argv():
    """Set the renderer screen env vars from --config BEFORE importing any
    torchwright_doom graph/pydoom module. Those modules build screen-sized vocab
    and read SCREEN_WIDTH at IMPORT time (apply_screen_env contract), so this
    MUST run first — otherwise everything renders at pydoom's default width 60
    regardless of --config (the bug that made e1m1 and e1m1_lowres produce
    identical token streams). Minimal YAML scrape to avoid importing the heavy
    config module (which itself triggers the vocab build)."""
    cfg = "configs/e1m1_lowres.yaml"
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            cfg = argv[i + 1]
        elif a.startswith("--config="):
            cfg = a.split("=", 1)[1]
    here = Path(__file__).resolve().parents[1]
    for cand in (
        Path(cfg),
        here / cfg,
        Path("/root") / cfg,
        Path("/root/torchwright_doom") / cfg,
    ):
        if cand.exists():
            text = cand.read_text()
            break
    else:
        return  # config not found yet (e.g. tiny-scene run); leave env as-is
    m = re.search(r"\bscale:\s*(\d+)", text)
    scale = int(m.group(1)) if m else 4
    d = re.search(r"\bdetail:\s*(\w+)", text)
    h = re.search(r"\bhud:\s*(\w+)", text)
    os.environ["TORCHWRIGHT_DOOM_RENDER_SCALE"] = str(scale)
    os.environ["TORCHWRIGHT_DOOM_SCREEN_WIDTH"] = str(320 // scale)
    os.environ["TORCHWRIGHT_DOOM_SCREEN_HEIGHT"] = str(200 // scale)
    os.environ["TORCHWRIGHT_DOOM_DETAIL"] = d.group(1) if d else "low"
    os.environ["TORCHWRIGHT_DOOM_HUD"] = "1" if (h and h.group(1) == "true") else "0"


_apply_screen_env_from_argv()

import torch

# NOTE (Phase 8c): this instrumentation harness still targets the deleted
# sinusoidal PosEncoding scheme — it hooks Attn.compute and classifies recency
# heads by the raw-counter column.  Under RoPE position is a rotation, so a
# rewrite must (a) build the graph with create_rope_config / GraphPast(rope=...),
# (b) classify a recency head by the presence of the global_position node
# (GraphPast.global_position) in its key_in leaves rather than a PosEncoding
# leaf, and (c) compute the position term as recency_scale·pos·cos((k-q)·theta)
# on the slowest rotated plane (apply_rope), not a raw qk-column dot.  Left
# runtime-broken (imports the removed pos_encoding module) pending that rewrite;
# the test suite does not import scripts.
#
# The recency-scheme DECISION this harness once informed is now settled: global
# recency at recency_scale=8 (= the old 8*counter) — see render_constants.py
# RECENCY_GAIN and the position-encoding-swap-research memo.  So the harness is
# no longer load-bearing for a decision; its remaining use is the 42k full-frame
# 0-flips / softmax-concentration replay, which is currently covered more
# authoritatively by the `test_flat_pixel_oracle` oracle asserts (cond near ±1)
# and the `make run COMPARE=1` render-vs-pydoom gate.  Rewrite is a nice-to-have.
from torchwright.graph import Concatenate
from torchwright.graph.attn import Attn
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.graph_debug import silenced_graph_asserts
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward
from torchwright_doom.vocab import NO_OP

_SUBMODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SUBMODULE_ROOT / "tests"))
from prefill_fixture import TINY_BSP_SCENE, row_index  # type: ignore  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_PATH = os.path.join(HERE, "position_attention_log.jsonl")  # overridable via --out
_MAX_STEPS = 8
CONTENT_TIE_EPS = 1.0  # content-logit window that counts as a "tie" to discriminate
TOPK = 16
_MAX_Q_PER_HEAD = 400  # subsample query positions per head on long frames
_MAX_CANDIDATES = 64  # cap candidate keys per record (bounds the no-match fallback)


# ---------------------------------------------------------------------------
# Static classification: for each Attn node, which qk columns are position?
# ---------------------------------------------------------------------------


def _leaf_rows_of_posencoding(key_in):
    """Return (counter_rows, trig_rows) absolute row indices in key_in that
    come from a PosEncoding leaf, or ([],[]) if key_in reads no PosEncoding."""
    leaves = key_in.flatten_inputs() if isinstance(key_in, Concatenate) else [key_in]
    counter_rows: list[int] = []
    trig_rows: list[int] = []
    row = 0
    for leaf in leaves:
        w = len(leaf)
        if isinstance(leaf, PosEncoding):
            counter_rows.append(row + leaf.counter_col)
            trig_rows.extend(
                range(row + leaf.trig_slice.start, row + leaf.trig_slice.stop)
            )
        row += w
    return counter_rows, trig_rows


def classify_attn(node: Attn):
    """Classify an Attn node's position usage. Returns a dict or None."""
    key_in = node.inputs[1]
    counter_rows, trig_rows = _leaf_rows_of_posencoding(key_in)
    if not counter_rows and not trig_rows:
        return None  # no position involved

    km = node.key_matrix  # (d_key_in, d_qk)
    d_qk = km.shape[1]

    def cols_touched(rows):
        if not rows:
            return set()
        sub = km[rows, :]
        return set((sub.abs().sum(dim=0) > 0).nonzero(as_tuple=True)[0].tolist())

    counter_cols = cols_touched(counter_rows)
    trig_cols = cols_touched(trig_rows)
    pos_cols = counter_cols | trig_cols
    content_cols = [c for c in range(d_qk) if c not in pos_cols]

    if counter_cols:
        kind = "recency_match"
    elif trig_cols and not content_cols:
        kind = "offset"
    elif trig_cols:
        kind = "offset_with_content"  # not expected in doom; flag if it appears
    else:
        return None

    return {
        "kind": kind,
        "content_cols": content_cols,
        "pos_cols": sorted(pos_cols),
        "d_qk": d_qk,
    }


# ---------------------------------------------------------------------------
# Logging hook around Attn.compute
# ---------------------------------------------------------------------------

_REGISTRY: dict[int, dict] = {}  # id(node) -> classification
_UID: dict[int, int] = {}  # id(node) -> stable small uid
_RECORDS: dict[tuple[int, int], dict] = {}  # (uid, query_pos) -> record (dedup)


def _install_logger():
    orig_compute = Attn.__dict__["compute"]

    def logging_compute(self, n_pos, input_values):
        info = _REGISTRY.get(id(self))
        if info is None:
            return orig_compute(self, n_pos, input_values)

        # Replicate the logit math (cheap relative to the value pass).
        query_in = self.inputs[0].compute(n_pos, input_values)
        key_in = self.inputs[1].compute(n_pos, input_values)
        qv = query_in @ self.query_matrix  # (n_pos, d_qk)
        kv = key_in @ self.key_matrix  # (n_pos, d_qk)
        dev = qv.device
        # value payload each key would deliver — used to test whether the 2nd
        # most-recent content-match carries a DIFFERENT value than the most
        # recent (i.e. whether recency is actually load-bearing, vs the value
        # being copied across the tied keys and recency not mattering).
        value_in = self.inputs[2].compute(n_pos, input_values)  # (n_pos, d_v)

        cc = info["content_cols"]
        pc = info["pos_cols"]
        content = (
            qv[:, cc] @ kv[:, cc].t() if cc else torch.zeros(n_pos, n_pos, device=dev)
        )
        posl = qv[:, pc] @ kv[:, pc].t()
        full = content + posl

        # Vectorise EVERYTHING on-device; pull only small per-row vectors to CPU.
        # A per-position Python loop with .item()/.tolist() would sync the GPU
        # hundreds of thousands of times — that was the original bottleneck.
        neg = torch.finfo(full.dtype).min
        ar = torch.arange(n_pos, device=dev)
        causal = ar[None, :] <= ar[:, None]  # (q, k): keep k <= q
        full_m = torch.where(causal, full, neg)
        content_m = torch.where(causal, content, neg)

        selected = full_m.argmax(dim=1)  # (n_pos,)
        sel_content = content.gather(1, selected[:, None]).squeeze(1)
        sel_pos = posl.gather(1, selected[:, None]).squeeze(1)

        uid = _UID[id(self)]
        kind = info["kind"]
        site = info["site"]

        # Which query positions to record (subsample long frames).
        if n_pos <= _MAX_Q_PER_HEAD:
            sample = list(range(n_pos))
        else:
            step = n_pos / _MAX_Q_PER_HEAD
            sample = sorted({int(i * step) for i in range(_MAX_Q_PER_HEAD)})

        sel_cpu = selected.cpu().tolist()
        selc_cpu = sel_content.cpu().tolist()
        selp_cpu = sel_pos.cpu().tolist()

        if kind == "recency_match":
            cmax = content_m.max(dim=1).values
            cargmax = content_m.argmax(dim=1)
            ties = ((content_m >= (cmax[:, None] - CONTENT_TIE_EPS)) & causal).sum(
                dim=1
            )
            below = torch.where(content_m < (cmax[:, None] - 1e-6), content_m, neg)
            second = below.max(dim=1).values
            cargmax_cpu = cargmax.cpu().tolist()
            ties_cpu = ties.cpu().tolist()
            cmax_cpu = cmax.cpu().tolist()
            second_cpu = second.cpu().tolist()
            # Candidate lists need full rows, but only for sampled q — gather those.
            sample_t = torch.tensor(sample, device=dev)
            content_s = content_m.index_select(0, sample_t).cpu()
            full_s = full_m.index_select(0, sample_t).cpu()
            k1s, k2s = [], []  # most-recent / 2nd-most-recent content-tie per sampled q
            for si, q in enumerate(sample):
                crow = content_s[si, : q + 1]
                frow = full_s[si, : q + 1]
                k = min(TOPK, q + 1)
                top_idx = set(torch.topk(frow, k).indices.tolist())
                cm = cmax_cpu[q]
                tie_idx = (
                    (crow >= cm - CONTENT_TIE_EPS).nonzero(as_tuple=True)[0].tolist()
                )
                # most-recent two content-ties (tie_idx is ascending positions)
                k1s.append(tie_idx[-1])
                k2s.append(tie_idx[-2] if len(tie_idx) >= 2 else -1)
                # Cap content-tie inclusion: in the no-match fallback EVERY causal
                # key is "tied", which would balloon the record to thousands of
                # entries (and the file to GBs). Those rows are excluded from
                # validation anyway (is_real). Keep at most _MAX_CANDIDATES of the
                # most-recent ties — recency only ever needs to split nearby keys.
                if len(tie_idx) > _MAX_CANDIDATES:
                    tie_idx = tie_idx[-_MAX_CANDIDATES:]
                top_idx.update(tie_idx)
                candidates = sorted(
                    ([int(j), round(float(crow[j]), 4)] for j in top_idx),
                    key=lambda kv2: -kv2[1],
                )
                sec = second_cpu[q]
                _RECORDS[(uid, q)] = {
                    "uid": uid,
                    "kind": kind,
                    "site": site,
                    "query_pos": q,
                    "n_pos_at_capture": n_pos,
                    "selected": sel_cpu[q],
                    "target_delta": q - sel_cpu[q],
                    "candidates": candidates,
                    "selected_content_logit": round(selc_cpu[q], 4),
                    "selected_pos_logit": round(selp_cpu[q], 4),
                    "content_argmax": cargmax_cpu[q],
                    "recency_decisive": cargmax_cpu[q] != sel_cpu[q],
                    "content_margin": None if sec <= neg / 2 else round(cm - sec, 4),
                    "n_content_ties": ties_cpu[q],
                }
            # Batched value-diff: does the 2nd most-recent content-tie deliver a
            # DIFFERENT value than the most recent? (max abs diff over the value
            # vector). None where there is no 2nd content-tie.
            k1t = torch.tensor(k1s, device=dev)
            k2t = torch.tensor(k2s, device=dev).clamp(min=0)
            vdiff = (
                (value_in.index_select(0, k1t) - value_in.index_select(0, k2t))
                .abs()
                .amax(dim=1)
            )
            vdiff_cpu = vdiff.cpu().tolist()
            for si, q in enumerate(sample):
                has2 = k2s[si] >= 0
                _RECORDS[(uid, q)]["tie2_value_maxdiff"] = (
                    round(vdiff_cpu[si], 6) if has2 else None
                )
                _RECORDS[(uid, q)]["tie2_value_differs"] = (
                    (vdiff_cpu[si] > 1e-4) if has2 else None
                )
        else:  # offset
            top2 = torch.topk(full_m, min(2, n_pos), dim=1).values  # (n_pos, <=2)
            runner = (
                top2[:, 1]
                if top2.shape[1] > 1
                else torch.full((n_pos,), neg, device=dev)
            )
            runner_cpu = runner.cpu().tolist()
            for q in sample:
                p_sel = selp_cpu[q]
                run = runner_cpu[q]
                _RECORDS[(uid, q)] = {
                    "uid": uid,
                    "kind": kind,
                    "site": site,
                    "query_pos": q,
                    "n_pos_at_capture": n_pos,
                    "selected": sel_cpu[q],
                    "target_delta": q - sel_cpu[q],
                    "delta": sel_cpu[q] - q,
                    "peak_logit": round(p_sel, 4),
                    "runner_logit": None if run <= neg / 2 else round(run, 4),
                    "peak_margin": None if run <= neg / 2 else round(p_sel - run, 4),
                }

        return orig_compute(self, n_pos, input_values)

    Attn.compute = logging_compute
    return orig_compute


# ---------------------------------------------------------------------------
# Representative token sequence: a real frame through rasterization
# ---------------------------------------------------------------------------


def frame_token_ids(config_rel="configs/e1m1_lowres.yaml", span=None):
    """Build the teacher-forcing token-id sequence for a real frame.

    Mirrors tests/scene/test_flat_pixel_oracle.py: load the WAD scene from the
    config, build the prefill prompt, append the pydoom drafter's golden AR
    tokens (BSP -> wall columns -> flat spans -> pixels -> DONE), and convert to
    W_EMBED row indices. `span` caps the AR body to `span` tokens past the seed
    (None = full frame). Pure-Python reference math, no GPU.
    """
    from torchwright_doom.inference.config import load_render_config
    from torchwright_doom.inference.wad_scene import (
        load_render_scene,
        pose_from_world,
        pydoom_scene_for,
    )
    from torchwright_doom.prompt.build import build_prompt
    import torchwright_doom.pydoom as pydoom

    # Resolve the submodule root robustly: locally it's parents[1], but under
    # `make modal-run` the repo is remounted (e.g. configs at /root/configs and
    # the package at /root/torchwright_doom). Pick the first candidate root that
    # actually contains the config file.
    candidates = [
        _SUBMODULE_ROOT,
        Path.cwd(),
        Path("/root"),
        Path("/root/torchwright_doom"),
    ]
    root = next((c for c in candidates if (c / config_rel).exists()), _SUBMODULE_ROOT)
    config_path = str(root / config_rel)
    config = load_render_config(config_path)
    scene = load_render_scene(config, base_dir=root)
    pose = pose_from_world(scene)
    py_scene = pydoom_scene_for(scene, pose)
    py_pose = py_scene.test_poses[0]

    prefill = list(build_prompt(scene.map_data, pose, asset_config=scene.asset_config))
    golden = list(pydoom.expected_ar_tokens(py_scene, py_pose))
    full = prefill + golden
    begin = len(prefill) - 1
    if span is not None:
        full = full[: min(begin + span, len(full) - 1) + 1]
    ids = [row_index(t.type, dict(t.values)) for t in full]
    return ids


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


_SITE: dict[int, str] = {}  # id(Attn) -> "file.py:line func (via handle)"


def _caller_site():
    """The renderer call site that built this Attn — skip the attention plumbing
    so the label points at the DOOM module, noting the handle it came through."""
    skip = {
        "attention_ops.py",
        "pos_encoding.py",
        "attention_handles.py",
        "attn.py",
        "position_attention_log.py",
        "past.py",
    }
    via = None
    for fr in inspect.stack():
        base = os.path.basename(fr.filename)
        if base in ("past.py", "attention_handles.py") and via is None:
            via = (
                fr.function
            )  # the wrapper: pick_most_recent / attend_to_offset / handle
        if base in skip:
            continue
        if "torchwright_doom" in fr.filename:
            lbl = f"{base}:{fr.lineno} {fr.function}"
            return f"{lbl} (via {via})" if via else lbl
    return via or "?"


def _install_site_capture():
    orig_init = Attn.__init__

    def init_with_site(self, *a, **k):
        orig_init(self, *a, **k)
        _SITE[id(self)] = _caller_site()

    Attn.__init__ = init_with_site
    return orig_init


def _ordered_nodes(root):
    """Deterministic node order via iterative DFS over node.inputs (lists, so
    stable). get_ancestor_nodes returns a SET whose iteration order varies by
    object hash across processes — using it for uid assignment makes uids
    non-reproducible. This makes uid stable so a record's head maps to a fixed
    site across runs."""
    seen, order, stack = set(), [], [root]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        order.append(n)
        stack.extend(n.inputs)
    return order


def _move_graph_to(nodes, device):
    """Move every node's weight tensors onto `device` so reference_eval runs
    there. reference_eval has no device knob: node.compute uses whatever device
    the stored matrices live on (CPU by default). PosEncoding synthesises fresh
    CPU tensors in get_pos_encoding regardless of attributes, so its compute is
    wrapped to push the result onto the device too."""
    for node in nodes:
        for k, v in list(node.__dict__.items()):
            if isinstance(v, torch.Tensor):
                setattr(node, k, v.to(device))
    orig_pe_compute = PosEncoding.__dict__["compute"]

    def pe_compute_on_device(self, n_pos, input_values):
        return orig_pe_compute(self, n_pos, input_values).to(device)

    PosEncoding.compute = pe_compute_on_device
    return orig_pe_compute


def build_and_run(teacher_force_ids=None, device="cpu"):
    """Run the rollout (or teacher-force a given token-id sequence) and log.

    teacher_force_ids: if given, a list[int] fed in ONE forward pass (every
        position logged at once). If None, free-runs the tiny BSP scene (stops
        at NO_OP, traversal only).
    device: where reference_eval runs. The oracle is O(n_pos^2); on a real frame
        use "cuda" (an idle A100 otherwise) — CPU is only sane for the tiny scene.
    """
    site_orig = _install_site_capture()
    try:
        iv = create_input("iv", TOKEN_VOCAB.layout.d_embed)
        pos = create_pos_encoding()
        nt = forward(iv, GraphPast(input_vec=iv, pos_encoding=pos), pos)
    finally:
        Attn.__init__ = site_orig

    nodes = _ordered_nodes(nt)  # deterministic -> uid maps to a fixed site
    uid = 0
    for node in nodes:
        if isinstance(node, Attn):
            info = classify_attn(node)
            if info is not None:
                info["site"] = _SITE.get(id(node), "?")
                _REGISTRY[id(node)] = info
                _UID[id(node)] = uid
                uid += 1
    print(f"position-using Attn nodes: {len(_REGISTRY)}  (device={device})")
    print("=== UID legend (uid: kind  site) ===")
    for nid, u in sorted(_UID.items(), key=lambda kv: kv[1]):
        info = _REGISTRY[nid]
        print(f"  {u}: {info['kind']}  {info['site']}")

    pe_orig = _move_graph_to(nodes, device) if device != "cpu" else None
    w_embed = W_EMBED.to(device)

    orig = _install_logger()
    try:
        if teacher_force_ids is not None:
            print(
                f"teacher-forcing {len(teacher_force_ids)} tokens on {device} "
                f"(reference_eval is O(n^2))...",
                flush=True,
            )
            rows = torch.stack([w_embed[i] for i in teacher_force_ids])
            with silenced_graph_asserts():
                reference_eval(nt, {"iv": rows}, len(teacher_force_ids))
        else:
            prefill_ids = [row_index(t, s) for t, s in TINY_BSP_SCENE]
            seq = list(prefill_ids)
            for _ in range(_MAX_STEPS):
                rows = torch.stack([w_embed[i] for i in seq])
                cache = reference_eval(nt, {"iv": rows}, len(seq))
                nxt = int(torch.argmax(cache[nt][-1] @ w_embed.t()).item())
                seq.append(nxt)
                if nxt == row_index(NO_OP, {}):
                    break
    finally:
        Attn.compute = orig
        if pe_orig is not None:
            PosEncoding.compute = pe_orig

    records = list(_RECORDS.values())
    records.sort(key=lambda r: (r["uid"], r["query_pos"]))
    with open(_OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} records to {_OUT_PATH}")
    return records


def summarize(records):
    by_uid = defaultdict(list)
    for r in records:
        by_uid[r["uid"]].append(r)
    print(f"\n{len(records)} records across {len(by_uid)} position-using heads")
    print(f"written to {_OUT_PATH}\n")

    rec_heads = [u for u, rs in by_uid.items() if rs[0]["kind"] == "recency_match"]
    off_heads = [u for u, rs in by_uid.items() if rs[0]["kind"] == "offset"]
    print(f"recency_match heads: {len(rec_heads)}   offset heads: {len(off_heads)}\n")

    # Restrict to REAL content selections (content discriminates: not every
    # causal key tied). The no-match fallback (uniform content -> recency picks
    # the current position) is excluded so the numbers reflect actual selections.
    def is_real(r):
        n_keys = r["query_pos"] + 1
        return n_keys >= 2 and r["n_content_ties"] < n_keys

    print("=== recency_match: real content selections, is position load-bearing? ===")
    print("  (decisive = content alone picks a DIFFERENT key than content+recency)")
    n_with_real = 0
    for u in rec_heads:
        real = [r for r in by_uid[u] if is_real(r)]
        if not real:
            continue
        n_with_real += 1
        decisive = [r for r in real if r["recency_decisive"]]
        deltas = [r["target_delta"] for r in real]
        # tightest gap between same-content-tier distractors that recency must split
        min_gap = None
        for r in real:
            if r["n_content_ties"] <= 1:
                continue
            ks = sorted(
                c[0]
                for c in r["candidates"]
                if c[1] >= r["selected_content_logit"] - CONTENT_TIE_EPS
            )
            gaps = [b - a for a, b in zip(ks, ks[1:])]
            if gaps:
                g = min(gaps)
                min_gap = g if min_gap is None else min(min_gap, g)
        print(
            f"  head {u}: {len(real)} real | decisive {len(decisive)} | "
            f"target_delta min/max {min(deltas)}/{max(deltas)} | "
            f"tightest tie gap: {min_gap}"
        )
    if not n_with_real:
        print("  (no real content selections in this run — frame didn't reach them)")

    print("\n=== offset: (delta) coverage + peak sharpness ===")
    for u in off_heads:
        rs = by_uid[u]
        deltas = sorted({r["delta"] for r in rs})
        margins = [r["peak_margin"] for r in rs if r["peak_margin"] is not None]
        qmax = max(r["query_pos"] for r in rs)
        print(
            f"  head {u}: deltas={deltas} | max query_pos {qmax} | "
            f"peak_margin min: {min(margins) if margins else 'n/a'}"
        )


# ---------------------------------------------------------------------------
# Validation harness: replay records against a candidate position scheme
# ---------------------------------------------------------------------------


def validate_recency(records, p_new, *, name="scheme", representative_only=True):
    """Re-score every recency_match record with content + p_new(query_pos,
    key_pos, n_pos) and check the argmax still equals the originally-selected
    key. Returns (n_checked, flips).

    representative_only: skip records in the no-match pure-recency fallback. In
        that fallback the content is UNIFORM across all causal keys (the marker/
        key the head looks for isn't present), so every key ties and recency
        picks the current position by default — not a real content selection. A
        record is representative iff content actually discriminates, i.e. not
        every causal key is in the top content tie.
    """
    flips = []
    checked = 0
    for r in records:
        if r["kind"] != "recency_match":
            continue
        if representative_only:
            n_keys = r["query_pos"] + 1
            if n_keys < 2 or r["n_content_ties"] >= n_keys:
                continue  # uniform content -> pure-recency fallback, not a real selection
        checked += 1
        n = r["n_pos_at_capture"]
        q = r["query_pos"]
        best_k, best_s = None, float("-inf")
        for k, c_logit in r["candidates"]:
            s = c_logit + p_new(q, k, n)
            if s > best_s:
                best_s, best_k = s, k
        if best_k != r["selected"]:
            flips.append(
                {
                    "uid": r["uid"],
                    "query_pos": q,
                    "expected": r["selected"],
                    "got": best_k,
                }
            )
    status = "OK" if not flips else f"FAIL ({len(flips)} flips)"
    print(f"[{name}] recency replay: {checked} representative selections -> {status}")
    for fl in flips[:10]:
        print(
            f"    head {fl['uid']} q={fl['query_pos']}: "
            f"expected {fl['expected']} got {fl['got']}"
        )
    return checked, flips


# Two candidate position terms, both drop-ins for the current 8*counter recency.


def p_old(query_pos, key_pos, n_pos):
    """The current scheme: linear in the raw counter, gain 8."""
    return 8.0 * key_pos


def p_coarse_10pct(query_pos, key_pos, n_pos):
    """A deliberately COARSE scheme: only distinguishes positions that differ by
    >~10% (log-bucketed at base 1.1). Two positions within 10% collapse to the
    same bucket -> equal position term -> recency cannot break their tie."""
    bucket = math.floor(math.log(key_pos + 1.0) / math.log(1.1))
    return 8.0 * bucket


def main():
    global _MAX_Q_PER_HEAD
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument(
        "--frame",
        action="store_true",
        help="teacher-force a real frame through rasterization",
    )
    ap.add_argument("--config", default="configs/e1m1_lowres.yaml")
    ap.add_argument(
        "--span",
        type=int,
        default=None,
        help="cap AR body to N tokens past the seed (--frame mode). "
        "Default None = full frame through DONE.",
    )
    ap.add_argument(
        "--tokens", help="JSON file with an explicit list[int] of token ids"
    )
    ap.add_argument(
        "--emit-stdout",
        action="store_true",
        help="also print JSONL between sentinels (for `make modal-run` "
        "where the on-disk file stays on the worker)",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="cpu or cuda (default: cuda if available). The oracle is "
        "O(n^2); run real frames on cuda.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build the frame token stream, print its length, and exit",
    )
    ap.add_argument(
        "--max-queries",
        type=int,
        default=400,
        help="max query positions logged per head (subsamples long frames)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="write JSONL here. On `make modal-run`, point at the mounted "
        "volume (/root/.cache/torchwright_doom/compiled/<name>.jsonl) and "
        "retrieve with `modal volume get torchwright-doom-render-cache <name>.jsonl`",
    )
    args = ap.parse_args()

    global _OUT_PATH
    if args.out:
        _OUT_PATH = args.out
    _MAX_Q_PER_HEAD = args.max_queries
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.tokens:
        tf = json.load(open(args.tokens))
    elif args.frame:
        tf = frame_token_ids(config_rel=args.config, span=args.span)
    else:
        tf = None

    if args.dry_run:
        n = len(tf) if tf is not None else 0
        print(
            f"dry-run: config={args.config} screen_width="
            f"{os.environ.get('TORCHWRIGHT_DOOM_SCREEN_WIDTH')} -> {n} tokens"
        )
        return

    records = build_and_run(teacher_force_ids=tf, device=device)
    summarize(records)

    # Validation first — it's the small, load-bearing result. The JSONL dump
    # comes last so that if a remote log stream truncates the big block, the
    # summary + validation are already captured.
    if args.validate:
        print("\n=== candidate-scheme validation (representative selections only) ===")
        validate_recency(records, p_old, name="old (8*counter)")
        validate_recency(records, p_coarse_10pct, name="coarse (>10% only)")

    if args.emit_stdout:
        # gzip+base64 so 10^4+ records survive `make modal-run` log capture
        # (raw JSONL gets rate-limited/truncated). Decode locally:
        #   awk '/JSONL-GZB64-BEGIN/{f=1;next}/JSONL-GZB64-END/{f=0}f' log \
        #     | base64 -d | gunzip > position_attention_log.jsonl
        import base64
        import gzip

        blob = "\n".join(json.dumps(r) for r in records).encode()
        b64 = base64.b64encode(gzip.compress(blob)).decode()
        print("===JSONL-GZB64-BEGIN===")
        for i in range(0, len(b64), 8000):
            print(b64[i : i + 8000])
        print("===JSONL-GZB64-END===")


if __name__ == "__main__":
    main()
