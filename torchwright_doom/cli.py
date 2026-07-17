"""Argument parsing and dispatch for the render-job CLI — nothing else.

Commands:

``python -m torchwright_doom compile --config job.yaml``
``python -m torchwright_doom run --config job.yaml --x 1056 --y -3616 --angle 64``
``python -m torchwright_doom compile-onnx-debug --config job.yaml``

Importing this module is safe before screen configuration: it imports only
the standard library at module level, the selected command implementation is
imported lazily after dispatch, and every implementation loads its config
and calls ``apply_screen_env`` before importing anything graph-reaching.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description="Compile and run YAML DOOM render jobs")
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("compile", help="compile config to a complete Phi-3 bundle")
    pc.add_argument("--config", required=True, dest="config_path")
    pc.add_argument("--verbose-compile", action="store_true", dest="verbose_compile")

    pod = sub.add_parser(
        "compile-onnx-debug", help="compile the explicit diagnostic ONNX artifact"
    )
    pod.add_argument("--config", required=True, dest="config_path")
    pod.add_argument("--verbose-compile", action="store_true", dest="verbose_compile")

    pr = sub.add_parser("run", help="render one pose from a YAML config")
    pr.add_argument("--config", required=True, dest="config_path")
    pr.add_argument("--x", type=float)
    pr.add_argument("--y", type=float)
    pr.add_argument("--angle", type=int)
    pr.add_argument("--viewz", type=float)
    # Run-knob defaults live in the config's ``run:`` section (run_config
    # resolves None there) — argparse must NOT restate them.
    pr.add_argument("--out-dir", default="out/render", dest="out_dir")
    pr.add_argument("--max-new-tokens", type=int, default=None, dest="max_new_tokens")
    pr.add_argument("--png", action="store_true")
    pr.add_argument("--compare", action="store_true", dest="compare_images")
    pr.add_argument("--png-zoom", type=int, default=8, dest="png_zoom")
    pr.add_argument(
        "--device",
        default="cpu",
        help="torch device for the HF model (cpu | cuda); the full frame "
        "needs a big GPU",
    )

    args = p.parse_args(argv)
    command = args.command
    values = vars(args)
    values.pop("command", None)
    # Exactly one business function is imported, lazily, per invocation.
    if command == "compile":
        from .bundle.build import compile_config

        compile_config(**values)
    elif command == "compile-onnx-debug":
        from .diagnostics.onnx import compile_onnx_debug_config

        compile_onnx_debug_config(**values)
    else:
        from .run import run_config

        run_config(**values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
