# scripts/ — what's what

Run any of these via `make modal-run MODULE=scripts.<name>` (GPU) or
locally where the docstring says CPU is fine. Three populations:

**Durable graph-measurement tools** (current, maintained):

- `analyze_forward_cost.py` — attribute the forward graph's depth
  (layers) and width (residual peak) to subsystems.
- `critical_path_extract.py` — the depth-driving scheduled-node chain.
- `residual_liveness.py` — live residual columns over layers.
- `widest_nodes.py` — the widest nodes in the compiled graph.
- `lane_census.py` — residual lane usage census.
- `compile_report.py` — compile summary (layers, width, schedule
  provenance).
- `consumer_profile.py` — conservative dense-fp32 weights + terminal-KV
  memory gate for a candidate consumer config; `--full-replay` measures exact
  trimmed layer shapes after the solve.

**Bundle and schedule debugging** (current, situational):

- `teacher_force_margins.py` — teacher-forced disagreement margins on a
  published bundle.
- `compile_onnx_debug_remote.py` — remote hand-over for the ONNX debug
  compile (`diagnostics/onnx.py`).
- `schedule_regression_probe.py` — two-schedule node diff for
  schedule-replay regressions.

**Regression-session probes** (kept deliberately as the durable record
of the July 2026 n_heads=32 recency regression; their companion working
docs were removed from the tree — see the git history around commit
`f7ececf`):

- `recency_rank_probe.py`, `recency_rank_envelope_model.py`,
  `recency_rank_scallop_model.py` — measurements behind the smoothed
  global-position fix (commit `4a0c3a2`).
