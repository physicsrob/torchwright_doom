"""Compare two render token dumps for token-level equivalence (Gate 1/3).

Usage: python -m scripts.compare_token_dumps <baseline.json> <candidate.json>
           [--allow-value-ties N]

Token-level is the honest static-cache equivalence claim: float logits may
differ at ulp level (cuBLAS reselects kernels at the static width S), but
every per-position argmax — prefill predictions AND rollout emissions —
must match exactly.  Exit 0 on identical, 1 with a first-diff report.

``--allow-value-ties N`` applies the mode=both contract
(cli.py::_assert_streams_equivalent) across dumps from DIFFERENT kernel
shapes (e.g. stride-bucketed vs full-stride runs): every diff must be a
pair of VALUE-type tokens (an fp32 adjacent-quantization-bin tie), lengths
must match, and at most N such ties are tolerated.  Any other diff — a
different token TYPE, a length diff, or a tie count over budget — still
exits 1.  Strict (N absent) remains the default.
"""

import argparse
import json


def _tokens(case: dict, key: str) -> list[str]:
    return [t["text"] for t in case[key]]


def _types(case: dict, key: str) -> list[str]:
    return [t["type"] for t in case[key]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument(
        "--allow-value-ties",
        type=int,
        default=None,
        dest="allow_value_ties",
        metavar="N",
        help="tolerate up to N positionwise VALUE-token diffs (fp32 "
        "bin-boundary ties across kernel shapes); any other diff is fatal",
    )
    args = ap.parse_args()
    base = json.load(open(args.baseline))["cases"][0]
    cand = json.load(open(args.candidate))["cases"][0]

    ok = True
    total_ties = 0
    keys = ("prefill_input_tokens", "predicted_next_tokens", "rollout_output_tokens")
    for key in keys:
        b, c = _tokens(base, key), _tokens(cand, key)
        if b == c:
            print(f"[compare] {key}: IDENTICAL ({len(b)} tokens)")
            continue
        if len(b) != len(c):
            # A length diff is fatal in both modes (a machinery bug shifts
            # every subsequent token; ties never change the length).
            ok = False
            print(f"[compare] {key}: LENGTH DIFF {len(b)} vs {len(c)}")
            n = min(len(b), len(c))
            for i in range(n):
                if b[i] != c[i]:
                    print(
                        f"[compare] {key}: FIRST DIFF at index {i}:\n"
                        f"  baseline : {b[i]!r}\n"
                        f"  candidate: {c[i]!r}"
                    )
                    break
            continue
        diffs = [i for i in range(len(b)) if b[i] != c[i]]
        if args.allow_value_ties is None:
            ok = False
            i = diffs[0]
            print(
                f"[compare] {key}: FIRST DIFF at index {i}:\n"
                f"  baseline : {b[i]!r}\n"
                f"  candidate: {c[i]!r}"
            )
            continue
        bt, ct = _types(base, key), _types(cand, key)
        non_value = [i for i in diffs if not (bt[i] == ct[i] == "value")]
        if non_value:
            ok = False
            i = non_value[0]
            print(
                f"[compare] {key}: NON-VALUE DIFF at index {i} "
                f"({bt[i]} vs {ct[i]}):\n"
                f"  baseline : {b[i]!r}\n"
                f"  candidate: {c[i]!r}"
            )
            continue
        total_ties += len(diffs)
        print(
            f"[compare] {key}: {len(diffs)} VALUE-token tie(s) at "
            f"{diffs[:6]}{'…' if len(diffs) > 6 else ''} ({len(b)} tokens)"
        )
    if args.allow_value_ties is not None and total_ties > args.allow_value_ties:
        ok = False
        print(
            f"[compare] {total_ties} value ties exceed the budget "
            f"--allow-value-ties {args.allow_value_ties}"
        )
    print(f"[compare] counts: base={base['counts']} cand={cand['counts']}")
    if ok and total_ties:
        print(f"[compare] RESULT: TOKEN-IDENTICAL up to {total_ties} value tie(s)")
    else:
        print("[compare] RESULT:", "TOKEN-IDENTICAL" if ok else "DIVERGED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
