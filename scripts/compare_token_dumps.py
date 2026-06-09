"""Compare two render token dumps for token-level equivalence (Gate 1/3).

Usage: python -m scripts.compare_token_dumps <baseline.json> <candidate.json>

Token-level is the honest static-cache equivalence claim: float logits may
differ at ulp level (cuBLAS reselects kernels at the static width S), but
every per-position argmax — prefill predictions AND rollout emissions —
must match exactly.  Exit 0 on identical, 1 with a first-diff report.
"""

import json
import sys


def _tokens(case: dict, key: str) -> list[str]:
    return [t["text"] for t in case[key]]


def main() -> int:
    base_path, cand_path = sys.argv[1], sys.argv[2]
    base = json.load(open(base_path))["cases"][0]
    cand = json.load(open(cand_path))["cases"][0]

    ok = True
    for key in ("prefill_input_tokens", "predicted_next_tokens", "rollout_output_tokens"):
        b, c = _tokens(base, key), _tokens(cand, key)
        if b == c:
            print(f"[compare] {key}: IDENTICAL ({len(b)} tokens)")
            continue
        ok = False
        if len(b) != len(c):
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
    print(f"[compare] counts: base={base['counts']} cand={cand['counts']}")
    print("[compare] RESULT:", "TOKEN-IDENTICAL" if ok else "DIVERGED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
