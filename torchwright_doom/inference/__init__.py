"""Transitional home of the render-job orchestration and bundle builder.

This package is being dissolved by the production-path cleanup
(plan_cleanup_v2.md); only ``cli`` remains — the argparse entrypoint
(``python -m torchwright_doom.inference``) whose ``run_config`` becomes root
``run.py`` and whose argparse shell becomes root ``cli.py``.

Everything else has moved to its lifecycle-stage package: the token contract
to ``tokenizer/`` (``rows``, ``codec``), the input side to ``prompt/scene``,
the output side to ``interpret/``, publication to ``bundle/``, the job spec
to root ``config.py`` / ``identity.py``, the graph authority to root
``model_graph.py``, and the ONNX diagnostics to ``diagnostics/onnx.py``.
"""
