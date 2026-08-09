"""The sole Doom inference program: portable stock-Hugging-Face generation.

This file is copied byte-identical to the root of every published Doom
bundle (``<bundle>/infer.py``) and executed there as a subprocess — by
anyone who downloads a bundle and by production render orchestration
(``run.py``) alike.  It is executed, never imported.  It intentionally
imports no TorchWright or ``torchwright_doom`` code: text enters through
the saved tokenizer, a stock ``Phi3ForCausalLM`` produces rows (a "row" is
one tokenizer id — one row of the tied embedding matrix, one sequence
position), and the only outputs are canonical integer ids plus their raw
tokenizer text.  No pixels are produced here: the bundle's standalone
tools (``tools/txt_to_png.py``; token protocol in ``PROTOCOL.md`` in the
source repo) decode those ids into a frame afterward and do no inference.
The bundle manifest's schema, field meanings, and completeness gate live
in ``torchwright_doom/bundle/manifest.py`` in the source repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# Multi-shard checkpoints otherwise load serially.  Keep these defaults in the
# portable program so downloaded bundles and production use the same loader.
os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "true")
os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "8")

import torch
import transformers
from transformers import TextGenerationPipeline, pipeline

_PROGRESS_INTERVAL_SECONDS = 15.0


def _canonical_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cuda_devices(model) -> list[torch.device]:
    return sorted(
        {parameter.device for parameter in model.parameters() if parameter.is_cuda},
        key=str,
    )


class _ProgressStreamer:
    """Report generation throughput without changing or collecting tokens."""

    def __init__(self, prompt_rows: int, max_new_tokens: int) -> None:
        self.prompt_rows = prompt_rows
        self.max_new_tokens = max_new_tokens
        self.started = time.monotonic()
        self.last_progress = self.started
        self.prefill_seconds: float | None = None
        self.finished: float | None = None
        self.generated_rows = 0
        self._saw_prompt = False

    def put(self, value: torch.Tensor) -> None:
        # GenerationMixin streams the complete prompt once before any emitted
        # row.  It is already accounted for separately in ``prompt_rows``.
        if not self._saw_prompt:
            self._saw_prompt = True
            return
        now = time.monotonic()
        if self.prefill_seconds is None:
            self.prefill_seconds = now - self.started
            print(
                f"[infer] prefill complete; rows={self.prompt_rows} "
                f"elapsed={self.prefill_seconds:.1f}s",
                flush=True,
            )
        self.generated_rows += value.numel()
        if now - self.last_progress >= _PROGRESS_INTERVAL_SECONDS:
            elapsed = now - self.started - self.prefill_seconds
            last_row = int(value.reshape(-1)[-1])
            print(
                f"[infer] decode rows={self.generated_rows}/{self.max_new_tokens} "
                f"total_position={self.prompt_rows + self.generated_rows} "
                f"elapsed={elapsed:.1f}s "
                f"rows/s={self.generated_rows / elapsed:.1f} "
                f"last_row={last_row}",
                flush=True,
            )
            self.last_progress = now

    def end(self) -> None:
        self.finished = time.monotonic()

    @property
    def decode_seconds(self) -> float:
        stopped = self.finished or time.monotonic()
        prefill = self.prefill_seconds or 0.0
        return stopped - self.started - prefill


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run stock Phi-3 Doom inference")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--output", type=Path, default=Path("out"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args(argv)

    # This file sits at the bundle root, so its own directory is the bundle.
    model_dir = (args.model or Path(__file__).resolve().parent).resolve()
    prompt_path = args.prompt or model_dir / "examples" / "e1m1_prompt.txt"
    manifest = json.loads((model_dir / "doom_bundle_manifest.json").read_text())
    if not manifest.get("validation", {}).get("complete"):
        raise ValueError("Doom bundle manifest is not complete")

    prompt_bytes = prompt_path.read_bytes()
    prompt_sha256 = _sha(prompt_bytes)
    bundled_prompt = prompt_sha256 == manifest["prompt"]["sha256"]

    load_t0 = time.monotonic()
    model_kwargs = {
        "attn_implementation": "eager",
        # Read each shard's bytes eagerly: deferring them to mmap page faults
        # stalls badly on network filesystems, and eager reads are harmless on
        # local disks.
        "disable_mmap": True,
    }
    generate: TextGenerationPipeline
    if args.device != "cpu":
        # Accelerate builds the skeleton on meta and dispatches each shard
        # directly to the target device.  This avoids a second full-model
        # ``model.to(cuda)`` pass through CPU-backed mmap pages.
        generate = pipeline(
            "text-generation",
            model=str(model_dir),
            dtype=torch.float32,
            model_kwargs=model_kwargs,
            device_map=args.device,
        )
    else:
        generate = pipeline(
            "text-generation",
            model=str(model_dir),
            dtype=torch.float32,
            model_kwargs=model_kwargs,
        )
    tokenizer = generate.tokenizer
    if tokenizer is None:
        raise RuntimeError("text-generation pipeline loaded without a tokenizer")
    model = generate.model
    model.eval()
    cuda_devices = _cuda_devices(model)
    for cuda_device in cuda_devices:
        # Reset after loading: the current allocation still includes all
        # weights, while the peak will additionally capture generation cache
        # and runtime workspace. This is the consumer-fit measurement.
        torch.cuda.reset_peak_memory_stats(cuda_device)
    attention_implementation = getattr(model.config, "_attn_implementation", None)
    if attention_implementation != "eager":
        # Eager is the implementation the published render was validated
        # under; fused kernels change fp accumulation order, and this check
        # keeps every run on the validated numerics.
        raise RuntimeError(
            "Doom inference requires eager attention, got "
            f"{attention_implementation!r}"
        )
    if (
        model.config.original_max_position_embeddings
        != model.config.max_position_embeddings
    ):
        raise RuntimeError(
            "default-RoPE Doom model has inconsistent original/max position "
            "capacity; GenerationMixin would discard its cache at the boundary"
        )
    load_seconds = time.monotonic() - load_t0

    prompt_text = prompt_bytes.decode("utf-8")
    encoded_prompt = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_device = next(model.parameters()).device
    prompt_ids = [int(row) for row in encoded_prompt.input_ids[0].tolist()]
    prompt_ids_sha256 = _sha(_canonical_json(prompt_ids))
    # Only the bundled prompt has a manifest row-id expectation; a custom
    # prompt is permitted, never verified, and recorded in the payload as
    # matches_bundled_prompt=false.
    if bundled_prompt and prompt_ids_sha256 != manifest["prompt"]["row_ids_sha256"]:
        raise ValueError("bundled prompt text does not reproduce its manifest rows")

    default_new = int(manifest["generation"]["max_new_tokens"])
    max_new = default_new if args.max_new_tokens is None else int(args.max_new_tokens)
    if max_new < 1:
        raise ValueError("max-new-tokens must be >= 1")
    if len(prompt_ids) + max_new > model.config.max_position_embeddings:
        raise ValueError("requested generation exceeds model position capacity")

    print(
        f"[infer] model ready in {load_seconds:.1f}s; prompt={len(prompt_ids)} "
        f"max_new_tokens={max_new} device={input_device}",
        flush=True,
    )
    generate_t0 = time.monotonic()
    progress = _ProgressStreamer(len(prompt_ids), max_new)
    print(f"[infer] generation started; max_new_tokens={max_new}", flush=True)
    with torch.inference_mode():
        records = generate(
            prompt_text,
            add_special_tokens=False,
            return_tensors=True,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            streamer=progress,
        )
    generate_seconds = time.monotonic() - generate_t0
    sequence = records[0]["generated_token_ids"]
    generated = [int(row) for row in sequence[len(prompt_ids) :]]
    for cuda_device in cuda_devices:
        torch.cuda.synchronize(cuda_device)
    cuda_memory = [
        {
            "device": str(cuda_device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(cuda_device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(cuda_device),
        }
        for cuda_device in cuda_devices
    ]
    prefill_seconds = progress.prefill_seconds or generate_seconds
    decode_seconds = progress.decode_seconds
    raw_text = tokenizer.decode(
        generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if not isinstance(raw_text, str):
        raise TypeError("tokenizer returned a batched decode for one row list")
    if tokenizer(raw_text, add_special_tokens=False)["input_ids"] != generated:
        raise ValueError("decoded output text does not round-trip to generated rows")

    args.output.mkdir(parents=True, exist_ok=True)
    emitted_ids_sha256 = _sha(_canonical_json(generated))
    stopped = bool(generated and generated[-1] == tokenizer.eos_token_id)
    payload = {
        "format": "torchwright_doom.output_ids.v1",
        "bundle": manifest.get("bundle_identity"),
        "compile_payload_sha256": manifest.get("compile_payload_sha256"),
        "row_vocab_fingerprint": manifest.get("row_vocab_fingerprint"),
        "prompt": {
            "sha256": prompt_sha256,
            "matches_bundled_prompt": bundled_prompt,
            "row_ids": prompt_ids,
            "row_ids_sha256": prompt_ids_sha256,
        },
        "emitted_row_ids": generated,
        "emitted_row_ids_sha256": emitted_ids_sha256,
        "generation": {
            "mode": "transformers_pipeline",
            "max_new_tokens": max_new,
            "termination_reason": "terminal" if stopped else "cap",
        },
        "timing_seconds": {
            "load": load_seconds,
            "prefill": prefill_seconds,
            "decode": decode_seconds,
            "generate": generate_seconds,
        },
        "attention_implementation": attention_implementation,
        "cuda_memory": cuda_memory,
        "transformers_version": transformers.__version__,
    }
    (args.output / "output.ids.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "output.txt").write_text(raw_text + "\n", encoding="utf-8")
    print(
        f"[infer] wrote {len(generated)} rows in {generate_seconds:.1f}s; "
        f"stopped={payload['generation']['termination_reason']}",
        flush=True,
    )
    for memory in cuda_memory:
        peak_allocated = int(memory["peak_allocated_bytes"])
        peak_reserved = int(memory["peak_reserved_bytes"])
        print(
            f"[infer] {memory['device']} peak allocated="
            f"{peak_allocated / 1024**3:.2f} GiB "
            f"reserved={peak_reserved / 1024**3:.2f} GiB",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
