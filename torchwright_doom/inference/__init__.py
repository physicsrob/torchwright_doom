"""Transitional home of the render-job orchestration and bundle builder.

This package is being dissolved by the production-path cleanup
(plan_cleanup_v2.md); what remains here moves out at the end:

- ``hf_bundle`` — the bundle builder (compile + staged publication);
  becomes ``bundle/build.py``. Manifest schema/validation already live in
  ``bundle/manifest.py``.
- ``cli`` — argparse entrypoint (``python -m torchwright_doom.inference``);
  ``run_config`` becomes root ``run.py`` and the argparse shell root
  ``cli.py``.

Everything else has moved to its lifecycle-stage package: the token contract
to ``tokenizer/`` (``rows``, ``codec``), the input side to ``prompt/scene``,
the output side to ``interpret/``, the job spec to root ``config.py`` /
``identity.py``, the graph authority to root ``model_graph.py``, and the
ONNX diagnostics to ``diagnostics/onnx.py``.

This package deliberately re-exports nothing: ``embedding`` builds the
screen-sized vocab AT IMPORT, so callers must ``apply_screen_env(config)``
before importing the modules that reach it (``tokenizer.rows``,
``model_graph``, ``hf_bundle``).
"""
