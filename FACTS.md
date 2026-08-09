# FACTS — the canonical numbers

Single source of truth for every load-bearing number in this project.
Other surfaces (README, CLAUDE.md, configs, the bundle model card, the
blog) state these values by copying from here or pointing here; if a
number elsewhere disagrees with this file, that number is wrong.
Update this file first, then the surfaces that quote it.

| Fact | Value | Source |
|---|---|---|
| Production frame | 320×200, low detail (160 rendered columns blitted 2 px wide) | `configs/e1m1.yaml` |
| Consumer frame | 80×50, low detail (40 rendered columns blitted 2 px wide) | `configs/e1m1_lowres.yaml` |
| Scene prompt | 3,614 rows (incl. `bos`) | 2026-07-15 production render |
| Rollout | 53,747 rows (`begin` … `done`) | 2026-07-15 production render log |
| Combined timeline | 57,361 rows (prompt + rollout) | arithmetic of the two above |
| Consumer rollout | 7,007 rows, with an 8,000-row generation cap (993 rows headroom) | exact pydoom protocol rollout for `configs/e1m1_lowres.yaml` |
| Consumer combined timeline | 10,621 rows expected; 11,614 rows at the configured cap | prompt + rollout / cap arithmetic |
| Consumer checkpoint | 70 layers; 34.09 GB / 31.75 GiB of fp32 weight shards | final bundle manifest |
| Consumer memory on A100-80GB | 43.18 GiB peak allocated, 43.48 GiB peak reserved; configured-cap conservative bound is 49.01 GiB (36.43 GiB weights + 12.41 GiB KV + 0.18 GiB cache-growth transient) | 2026-08-09 full pipeline render / `scripts/consumer_profile.py` |
| Consumer decode time | 338.3 s decode; 342.9 s generation including 4.6 s prefill; 576.0 s network-volume model load on one A100-80GB | 2026-08-09 full pipeline render |
| Consumer accuracy vs pydoom oracle | 100.0% coverage, 100.0% within-option color, 93.9% exact | 2026-08-09 full pipeline render, 3,964 generated/reference pixels |
| Full-checkpoint memory on B200 | 147.38 GiB peak allocated, 151.00 GiB peak reserved | 2026-08-09 full pipeline render |
| Decode time | 2,383.5 s greedy decode (39.7 min, 22.55 rows/s average) on one B200; 2,386.7 s generation including 3.2 s prefill | 2026-08-09 full pipeline render |
| End-to-end time | 2,528.1 s (42.1 min) including 141.4 s network-volume model load | 2026-08-09 full pipeline render |
| Frame rate | ≈0.0004 fps | derived from decode time |
| Checkpoint | 85.87 GB / 79.97 GiB dense fp32, sharded safetensors, stock `Phi3ForCausalLM` (no custom code, no `trust_remote_code`) | final bundle manifest |
| Layers | 38 | final bundle manifest; existing schedule replayed from cache without a re-solve |
| Accuracy vs pydoom oracle | 100.0% coverage, 99.9% within-option color, 96.7% exact | 2026-08-09 full pipeline render, 63,490 generated/reference pixels |
| Map / scene | E1M1, fixed world-space region `{x1: 627.2, y1: -3760.0, x2: 1395.2, y2: -2800.0}` (declared once in the config; view-independent) | `configs/e1m1.yaml` |
| WAD | `doom1.wad` — the freely redistributable shareware DOOM 1.9 WAD (4,196,020 bytes) | repo root |

Measurement provenance: the 2026-08-09 release render is the 320×200
`configs/e1m1.yaml` run through the stock Transformers pipeline on a Modal
B200. Historical figures from the retired ONNX/windowed-KV runtime (160×100,
~25,000 tokens, ~0.002 fps) describe a different engine and must not be quoted
for the current system. The consumer profile
prices the configured global head/MLP caps; the completed bundle manifest and
full render supersede it if their exact layer geometry differs. The consumer's
70 layers are a selected release depth, not a claimed global minimum: CP-SAT
proved 70 only for the explicit `n_layers >= 70` selection run, and the
published schedule records `selected_is_optimal: false`.
