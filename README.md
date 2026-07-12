# torchwright_doom

DOOM rendering and game graph compilation, built on [torchwright](https://github.com/physicsrob/torchwright). Flagship project of the [torchdoom](https://github.com/physicsrob/torchdoom) umbrella.

## What this is

This package builds a **computation graph** that the `torchwright` compiler
turns into a **transformer**, which then renders DOOM **autoregressively** —
one discrete input token in, one output token out, per step, the same loop
that drives any chat model. Output tokens carry pixel information; the host
copies each output token to the next input and blits pixels to the screen.

The production artifact is a stock Hugging Face `Phi3ForCausalLM`, compiled
directly from the Doom graph into sharded fp32 safetensors. Both model and the
data-only semantic WordLevel tokenizer load through ordinary auto classes with
no custom model/tokenizer code and no `trust_remote_code=True`. ONNX remains an
explicit diagnostic backend and is not a render or publication input.

The **dumb-host principle** governs everything: all rendering logic — wall
selection, visibility, distance, texture lookup, compositing — lives inside
the transformer. The host only feeds tokens and writes pixels; it does no
geometry, sorting, or arithmetic. (See `CLAUDE.md`.)

## Where to start

- **Entry point:** the per-token forward pass is `render_main.forward`
  (`torchwright_doom/render_main.py`). It builds the read side (decode the
  input token + consult static map facts), publishes the write-side
  protocol owners, builds each dispatch branch's next-token, and selects one
  by the current token's type.
- **Reading path** (one `forward()` pass, read side → write side):
  `vocab` / `tokens` → `embedding` / `extract` → `scene_tokens` /
  `scene_headers` / `scene_index` / `scene_facts` (static read side) →
  `protocol_tokens` / `protocol_registry` (the dispatch table) →
  `render_main.forward` (assembly) → the write side: `bsp_traversal`
  (R_RenderBSPNode) → `seg_projection` → the `wall_*` / `visplane_*` /
  `flat_*` rasterizers → the pixel pass.
- **Prefill pipeline** (WAD → tokens the model reads before autoregression):
  `doom1.wad` → `prompt/wad.py` (raw `MapData`) → `prompt/subset.py`
  (bbox-sliced, renumbered, mean-centred) → `prompt/build.py` (`list[Token]`)
  → `tokenizer/rows.py` (row indices) → the model. Production entry:
  `prompt/scene.prefill_rows_for`.

## Docs

- **`CLAUDE.md`** — full module layout, the production HF runtime, and
  the graph-debugging tool sequence.
- **`GLOSSARY.md`** — plain-English definitions of the coined vocabulary
  (carrier, head, marker, owner, subcontext, visplane, flat, …).
- **`PROTOCOL.md`** — the pixel protocol: the exact per-frame token
  sequence (prefill + every AR phase), in the readable-surface token names.
- **`protocol_registry.render_protocol_table()`** — the generated table of
  the token protocol (every token type, its phase, role, and dispatch
  wiring), for a top-down view of the AR protocol.

## Running

`make compile` creates the complete validated Phi-3 bundle on Modal
(publication, `bundle/`). `make run` resolves that same bundle and executes
its exact bundle-root `infer.py` on the configured GPU (portable inference);
it never compiles, converts, or uses a private generation path inside the
renderer. Everything after the subprocess is interpretation (`interpret/`).
`configs/e1m1.yaml` is the sole full-resolution publication configuration,
while `configs/e1m1_lowres.yaml` is retained for preview and validation.

The published bundle's claim-bearing load path is ordinary Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(bundle)
model = AutoModelForCausalLM.from_pretrained(
    bundle,
    attn_implementation="eager",
    device_map="cuda",
)
```

The bundle contains its executable E1M1 text prompt (`examples/e1m1_prompt.txt`)
and `infer.py` at the bundle root. That isolated script is the only inference
program used by both downloaded and production renders. It writes canonical
integer row ids and their raw standard-tokenizer text. `tools/pretty_text.py`
formats that text for reading, and `tools/txt_to_png.py` independently turns
the same text into a frame using only cursor bookkeeping, palette lookup, and
last-write-wins blitting:

```bash
python infer.py --model . --prompt examples/e1m1_prompt.txt --output out
python tools/pretty_text.py --input out/output.txt --output out/output.pretty.txt
python tools/txt_to_png.py  --input out/output.txt --output out/frame.png
```
