# FACTS — the canonical numbers

Single source of truth for every load-bearing number in this project.
Other surfaces (README, CLAUDE.md, configs, the bundle model card, the
blog) state these values by copying from here or pointing here; if a
number elsewhere disagrees with this file, that number is wrong.
Update this file first, then the surfaces that quote it.

| Fact | Value | Source |
|---|---|---|
| Production frame | 320×200, low detail (160 rendered columns blitted 2 px wide) | `configs/e1m1.yaml` |
| Preview frame | 160×100, low detail (80 rendered columns) | `configs/e1m1_lowres.yaml` |
| Scene prompt | 3,614 rows (incl. `bos`) | 2026-07-15 production render |
| Rollout | 53,747 rows (`begin` … `done`) | 2026-07-15 production render log |
| Combined timeline | 57,361 rows (prompt + rollout) | arithmetic of the two above |
| Decode time | 2,498 s greedy decode (~42 min) on one B200; throughput starts ~38 tok/s and decays as the KV cache grows | 2026-07-15 production render log |
| End-to-end time | 2,753 s (~46 min) including model load | 2026-07-15 production render log |
| Frame rate | ≈0.0004 fps | derived from decode time |
| Checkpoint | ~98 GB dense fp32, sharded safetensors, stock `Phi3ForCausalLM` (no custom code, no `trust_remote_code`) | bundle manifest / `infer.py` |
| Layers | **set at publication** — the published bundle's exact `n_layers` goes here (the schedule solver lands 35–40 run to run; never quote a layer count without it) | pending publication compile |
| Accuracy vs pydoom oracle | ~99.99% within-option color, 96.5% exact | 2026-07-15 production render, `make run COMPARE=1` |
| Map / scene | E1M1, fixed world-space region `{x1: 627.2, y1: -3760.0, x2: 1395.2, y2: -2800.0}` (declared once in the config; view-independent) | `configs/e1m1.yaml` |
| WAD | `doom1.wad` — the freely redistributable shareware DOOM 1.9 WAD (4,196,020 bytes) | repo root |

Measurement provenance: the 2026-07-15 production render is the
320×200 `configs/e1m1.yaml` run on Modal (B200) whose rollout the blog
artifacts replay. Historical figures from the retired ONNX/windowed-KV
runtime (160×100, ~25,000 tokens, ~0.002 fps) describe a different
engine and must not be quoted for the current system.
