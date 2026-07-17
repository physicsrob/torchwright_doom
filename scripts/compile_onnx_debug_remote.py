"""One-shot remote diagnostic compile — the Modal path for compile-onnx-debug.

The diagnostic cache key embeds git SHAs that are unresolvable inside a
Modal container (no ``.git``), so ``canonical_onnx_debug_payload`` raises
there by design: the key and the sidecar's ``compile_payload`` must be
computed LOCALLY and handed over (see ``compile_onnx_debug_cached``).
This script is both halves of that hand-over:

Local (where .git is available) — print the remote ARGS::

    uv run python scripts/compile_onnx_debug_remote.py \
        --config configs/e1m1.yaml --print-args

Remote (paste the printed string; big CP-SAT solves want a long timeout)::

    MODAL_RUN_TIMEOUT=14400 make modal-run \
        MODULE=scripts.compile_onnx_debug_remote ARGS="<printed>"

The artifact lands on the ONNX-debug volume under the printed cache key;
``blog/pieces/doom`` pulls the sidecar down with
``extract_floorplan.py --fetch-sidecar`` (or ``--latest``).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, dest="config_path")
    ap.add_argument(
        "--print-args",
        action="store_true",
        help="LOCAL mode: compute the cache key + compile payload from the "
        "working tree and print the ARGS string for the remote run",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="REMOTE mode: volume-backed cache dir (computed locally)",
    )
    ap.add_argument(
        "--payload-b64",
        default=None,
        help="REMOTE mode: base64 canonical_onnx_debug_payload (computed locally)",
    )
    ap.add_argument("--verbose-compile", action="store_true", dest="verbose_compile")
    args = ap.parse_args(argv)

    from torchwright_doom.config import (
        apply_screen_env,
        load_render_config,
        resolve_wad_path,
    )

    config_path = Path(args.config_path)
    config = load_render_config(config_path)
    apply_screen_env(config)

    if args.print_args:
        from torchwright_doom.diagnostics.onnx import canonical_onnx_debug_payload
        from torchwright_doom.identity import cache_key_from_payload

        wad_path = resolve_wad_path(config, base_dir=config_path.parent)
        payload = canonical_onnx_debug_payload(config, wad_path)
        key = cache_key_from_payload(payload)
        b64 = base64.b64encode(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).decode("ascii")
        remote_cache = f"/root/.cache/torchwright_doom/onnx_debug/{key}"
        print(
            f"--config {args.config_path} --cache-dir {remote_cache} "
            f"--payload-b64 {b64} --verbose-compile"
        )
        return 0

    if not args.cache_dir or not args.payload_b64:
        ap.error("remote mode needs --cache-dir and --payload-b64 (see --print-args)")
    from torchwright_doom.diagnostics.onnx import compile_onnx_debug_cached

    payload = json.loads(base64.b64decode(args.payload_b64))
    cache_dir = compile_onnx_debug_cached(
        config,
        base_dir=config_path.parent,
        verbose=args.verbose_compile,
        cache_dir=args.cache_dir,
        compile_payload=payload,
    )
    print(f"[remote] diagnostic artifact at {cache_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
