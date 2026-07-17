"""Publication: compile_config orchestration, compile, staged publication.

Manifest schema, hashing, and completeness validation live in ``manifest``;
the copied bundle layout in ``layout``. This module owns ``compile_config``
(cache probe + Modal-handed payloads), the stock Phi-3 compilation via
``model_graph.build_graph`` and torchwright, model-config stamping, staged
model/tokenizer smoke validation (through the shipped pure-stdlib formatter
``portable/pretty_text.py``), and the rollback-protected two-rename
publication.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from ..config import (
    RenderConfig,
    apply_screen_env,
    load_render_config,
    resolve_wad_path,
)
from ..identity import (
    canonical_compile_payload,
    hf_bundle_cache_dir,
    validate_compile_payload,
)
from .manifest import (
    MANIFEST_NAME,
    candidate_manifest,
    is_complete_hf_bundle,
    validate_bundle_manifest,
)


def compile_config(
    *,
    config_path: str | Path,
    verbose_compile: bool = False,
    cache_dir: str | Path | None = None,
    compile_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile ``config_path`` into a complete published bundle on a cache
    miss. ``cache_dir`` / ``compile_payload`` exist for the Modal path: the
    key embeds git SHAs unresolvable inside a container, so the local
    entrypoint computes both and hands them over."""
    config_path = Path(config_path)
    config = load_render_config(config_path)
    apply_screen_env(config)
    wad_path = resolve_wad_path(config, base_dir=config_path.parent)
    payload = compile_payload or canonical_compile_payload(config, wad_path)
    destination = (
        Path(cache_dir)
        if cache_dir is not None
        else hf_bundle_cache_dir(config, wad_path)
    )
    if is_complete_hf_bundle(destination, expected_payload=payload):
        print(f"[compile] direct-HF cache hit {destination}", flush=True)
        return {"cache_dir": str(destination), "cache_hit": True}
    print(f"[compile] direct-HF cache miss {destination}", flush=True)
    report = compile_phi3_bundle(
        config,
        wad_path=wad_path,
        destination=destination,
        compile_payload=payload,
        verbose=verbose_compile,
    )
    provenance = report.manifest["schedule"]
    # Schedule-provenance fields, defined by torchwright's ScheduleProvenance
    # (torchwright/compiler/token_model.py): selected_origin = where the
    # winning schedule came from (fresh solve vs. cache), delivery = how this
    # compile obtained it, selected_objective = the selected schedule's
    # solver objective value.
    print(
        f"[compile] {report.n_layers} layers "
        f"(selected={provenance.get('selected_origin')}, "
        f"delivery={provenance.get('delivery')}, "
        f"objective={provenance.get('selected_objective')}) -> {destination}",
        flush=True,
    )
    return {**report.to_dict(), "cache_dir": str(destination), "cache_hit": False}


@dataclass(frozen=True)
class BundleReport:
    destination: str
    n_layers: int
    vocab_size: int
    manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _remove(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


@contextmanager
def _outer_bundle_transaction(destination: str | Path) -> Iterator[Path]:
    """Stage a complete bundle beside its destination and publish with rollback.

    Staging and final paths share a filesystem so each rename is atomic and a
    failed second rename can restore the previous destination. The two-rename
    replacement is NOT reader-visible atomic — there is a short gap after the
    old destination moves aside. Production assumes one publisher per cache
    key; publication is not serialized against concurrent publishers.
    """
    final = Path(destination).absolute()
    final.parent.mkdir(parents=True, exist_ok=True)
    root = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.doom-staging-", dir=final.parent)
    )
    bundle = root / "bundle"
    backup = final.parent / f".{final.name}.previous-{root.name}"
    try:
        yield bundle
        if not bundle.is_dir():
            raise RuntimeError("Doom bundle transaction produced no staged directory")
        if final.exists():
            os.replace(final, backup)
        try:
            os.replace(bundle, final)
        except BaseException:
            if backup.exists():
                os.replace(backup, final)
            raise
        _remove(backup)
    except BaseException:
        if backup.exists() and not final.exists():
            os.replace(backup, final)
        raise
    finally:
        _remove(root)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _stamp_model_config(
    bundle: Path,
    *,
    config: RenderConfig,
    vocab_fingerprint: str,
    eos_row: int,
) -> None:
    path = bundle / "config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pad_token_id"] = int(eos_row)
    # Phi3's GenerationMixin resets its cache at this field even for default
    # RoPE, where no short/long-RoPE switch exists.  The compiled model has one
    # position regime spanning the full configured context.
    payload["original_max_position_embeddings"] = config.model.max_seq_len
    payload["doom_screen_config"] = {
        "width": config.screen[0],
        "height": config.screen[1],
        "scale": config.model.scale,
        "detail": config.model.detail,
        "hud": config.model.hud,
    }
    payload["doom_vocab_fingerprint"] = vocab_fingerprint
    _write_json(path, payload)


def _validate_complete_staged_bundle(
    bundle: Path, config: RenderConfig, prompt_rows: list[int]
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    # Staged validation smoke-tests the exact shipped pure-stdlib formatter
    # (portable/pretty_text.py), not the project wrapper — publication has
    # no edge into interpret/.
    from ..portable.pretty_text import DoomTextFormatter

    manifest = validate_bundle_manifest(bundle, allow_incomplete=True)
    tokenizer = AutoTokenizer.from_pretrained(bundle)
    if len(tokenizer) != manifest["vocab_size"]:
        raise ValueError("tokenizer width differs from Doom manifest")
    if tokenizer.bos_token_id != manifest["bos_token_id"]:
        raise ValueError("tokenizer BOS id differs from Doom manifest")
    if tokenizer.eos_token_id != manifest["eos_token_id"]:
        raise ValueError("tokenizer EOS id differs from Doom manifest")
    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        raise ValueError("tokenizer padding does not reuse DONE/EOS")
    prompt_text = (bundle / manifest["prompt"]["path"]).read_text()
    actual_prompt = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if actual_prompt != prompt_rows:
        raise ValueError("bundled prompt text does not reproduce build-time rows")
    formatter = DoomTextFormatter.from_bundle(bundle, allow_incomplete=True)
    raw_prompt = tokenizer.decode(
        prompt_rows, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if not isinstance(raw_prompt, str):
        raise TypeError("tokenizer returned a batched decode for one prompt row list")
    if (
        formatter.rows_from_raw_text(
            formatter.parse_pretty_text(formatter.format_text(raw_prompt))
        )
        != prompt_rows
    ):
        raise ValueError("DoomTextFormatter changed bundled prompt row identity")

    model = AutoModelForCausalLM.from_pretrained(
        bundle, attn_implementation="eager", dtype=torch.float32
    ).eval()
    if type(model).__name__ != "Phi3ForCausalLM":
        raise ValueError(f"expected Phi3ForCausalLM, loaded {type(model).__name__}")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Doom bundle contains a non-fp32 model parameter")
    if any(name.endswith(".bias") for name, _ in model.named_parameters()):
        raise ValueError("Doom Phi-3 bundle unexpectedly contains projection biases")
    expected = {
        "hidden_size": config.model.d,
        "head_dim": config.model.d_head,
        "max_position_embeddings": config.model.max_seq_len,
        "original_max_position_embeddings": config.model.max_seq_len,
        "vocab_size": manifest["vocab_size"],
        "bos_token_id": manifest["bos_token_id"],
        "eos_token_id": manifest["eos_token_id"],
        "pad_token_id": manifest["pad_token_id"],
        "num_hidden_layers": manifest["compile"]["n_layers"],
        "rms_norm_eps": 1e-5,
    }
    for name, value in expected.items():
        if getattr(model.config, name) != value:
            raise ValueError(
                f"stock Phi-3 config {name}={getattr(model.config, name)!r} "
                f"does not match expected {value!r}"
            )
    if model.config.hidden_act != "silu":
        raise ValueError("Doom production model is not SwiGLU/SiLU")
    hidden_cap = config.model.d_hidden or config.model.d
    if not (0 < model.config.intermediate_size <= hidden_cap):
        raise ValueError("stock Phi-3 intermediate width exceeds the Doom compile cap")
    partial = model.config.rope_parameters.get("partial_rotary_factor")
    expected_partial = (config.model.d_rot or config.model.d_head) / config.model.d_head
    if partial != expected_partial:
        raise ValueError("stock Phi-3 partial-RoPE setting differs from the Doom graph")
    # token.v6 tie (the tied token contract; see GLOSSARY.md: token.v6): one
    # serialized token table serves lookup and readout — the config declares
    # the tie, the loaded parameters share storage, and the index carries no
    # separate lm_head tensor.
    if model.config.tie_word_embeddings is not True:
        raise ValueError("Doom Phi-3 bundle must set tie_word_embeddings (token.v6)")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise ValueError(
            "loaded Doom model's lm_head is not storage-tied to embed_tokens"
        )
    weight_map = json.loads((bundle / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if "model.embed_tokens.weight" not in weight_map:
        raise ValueError("Doom bundle index lacks model.embed_tokens.weight")
    if "lm_head.weight" in weight_map:
        raise ValueError("Doom bundle serializes an untied lm_head.weight")
    cache = DynamicCache()
    first = torch.tensor([[manifest["bos_token_id"]]], dtype=torch.long)
    with torch.no_grad():
        result = model(
            input_ids=first,
            past_key_values=cache,
            cache_position=torch.tensor([0]),
            use_cache=True,
        )
        second = model(
            input_ids=torch.tensor([[manifest["eos_token_id"]]], dtype=torch.long),
            past_key_values=result.past_key_values,
            cache_position=torch.tensor([1]),
            use_cache=True,
        )
    if (
        result.logits.shape[-1] != manifest["vocab_size"]
        or second.logits.shape[-1] != manifest["vocab_size"]
    ):
        raise ValueError("stock Phi-3 smoke test produced the wrong logit width")
    del model


def compile_phi3_bundle(
    config: RenderConfig,
    *,
    wad_path: str | Path,
    destination: str | Path,
    compile_payload: dict[str, Any],
    verbose: bool = False,
) -> BundleReport:
    """Compile and publish one complete production Doom bundle (with rollback)."""
    if config.map.upper() != "E1M1":
        raise ValueError(
            "the production HF bundle contract currently supports E1M1 only"
        )
    wad_path = Path(wad_path)
    validate_compile_payload(compile_payload, config, wad_path)

    from ..model.constants import SCREEN_HEIGHT, SCREEN_WIDTH

    if (SCREEN_WIDTH, SCREEN_HEIGHT) != config.screen:
        raise RuntimeError(
            f"screen dims {SCREEN_WIDTH}x{SCREEN_HEIGHT} were imported before "
            f"this config's {config.screen[0]}x{config.screen[1]} was applied; "
            "call apply_screen_env(config) before importing graph/tokenizer modules"
        )

    from torchwright.compiler import CompileProfile, compile_hf_bundle

    from .layout import write_bundle_layout, write_model_card
    from ..model_graph import build_graph
    from ..prompt.scene import load_render_scene, pose_from_world, prefill_rows_for
    from ..tokenizer.codec import raw_text_from_rows
    from ..tokenizer.freeze import write_frozen_data
    from ..tokenizer.identity import vocab_fingerprint
    from ..tokenizer.standard import (
        build_standard_tokenizer,
        canonical_words,
        doom_special_tokens,
        ordered_words_sha256,
    )

    next_token, _rope, embedding, asset_banks = build_graph(
        d_head=config.model.d_head,
        max_positions=config.model.max_seq_len,
        d_rot=config.model.d_rot,
        asset_config=config.asset_config(),
        wad_path=wad_path,
    )
    compiler_vocab = list(embedding.tokenizer.vocab)
    special = doom_special_tokens(compiler_vocab)
    words = canonical_words(config.asset_config())
    tokenizer = build_standard_tokenizer(config.asset_config(), words=words)

    scene = load_render_scene(config, base_dir=wad_path.parent)
    pose = pose_from_world(scene)
    prompt_rows = prefill_rows_for(scene, pose)
    prompt_text = raw_text_from_rows(words, prompt_rows) + "\n"

    with _outer_bundle_transaction(destination) as stage:
        report = compile_hf_bundle(
            next_token,
            embedding,
            stage,
            d=config.model.d,
            d_head=config.model.d_head,
            n_heads=config.model.n_heads,
            max_seq_len=config.model.max_seq_len,
            max_layers=config.model.max_layers,
            optimize=config.model.optimize,
            d_hidden=config.model.d_hidden,
            trim_heads=config.model.trim_heads,
            # RMSNorm pinned-constant exponent: the norm is kept a bit-exact
            # identity by reserved residual columns holding large power-of-two
            # constants. 63 is the largest fp32-feasible exponent; its
            # data-energy budget (~2^102) clears the doom graph's residual
            # energy bound (~2^99.6, fixed-point coordinates) with ~5x margin,
            # where torchwright's default was tuned on far smaller graphs.
            rms_norm_const_exp=63,
            architecture=CompileProfile.PHI3,
            bias=False,
            bos_token=special.bos_compiler_string,
            eos_token=special.eos_compiler_string,
            verbose=verbose,
            add_bos_token=False,
            write_tokenizer=False,
        )
        tokenizer.save_pretrained(stage)
        fingerprint = vocab_fingerprint()
        write_frozen_data(
            stage,
            asset_config=config.asset_config(),
            palette=asset_banks.playpal,
            origin=scene.origin,
        )
        _stamp_model_config(
            stage,
            config=config,
            vocab_fingerprint=fingerprint,
            eos_row=special.eos_row,
        )
        written_layout = write_bundle_layout(stage, prompt_text=prompt_text)
        write_model_card(stage, config)
        prompt_path = next(
            path for path in written_layout if path.name == "e1m1_prompt.txt"
        )
        manifest = candidate_manifest(
            bundle=stage,
            config=config,
            wad_path=wad_path,
            compile_payload=compile_payload,
            report=report,
            prompt_rows=prompt_rows,
            prompt_path=prompt_path,
            vocab_size=len(words),
            row_vocab_fingerprint=fingerprint,
            tokenizer_vocab_sha256=ordered_words_sha256(words),
            bos_row=special.bos_row,
            eos_row=special.eos_row,
            origin=scene.origin,
        )
        _write_json(stage / MANIFEST_NAME, manifest)
        _validate_complete_staged_bundle(stage, config, prompt_rows)
        manifest["validation"]["complete"] = True
        _write_json(stage / MANIFEST_NAME, manifest)

    return BundleReport(
        destination=str(destination),
        n_layers=int(report.n_layers),
        vocab_size=len(words),
        manifest=manifest,
    )
