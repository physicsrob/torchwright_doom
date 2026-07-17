"""Write a Doom bundle's executable layout: root ``infer.py``, ``tools/``, prompt.

The bundle root carries a byte-identical copy of the sole inference program
(``torchwright_doom/infer.py``); the standalone artifact consumers ship under
``tools/`` from their ``portable/`` sources; ``examples/`` contains data only
(the executable prompt). Byte-identity of every copied file against its
source is enforced by the bundle layout gate (``tests/bundle/test_layout.py``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..config import RenderConfig

_TOOLS_README = """# Reproduce the Doom render

Run stock Hugging Face inference first, from the bundle root:

```bash
python infer.py --model . --prompt examples/e1m1_prompt.txt --output out
```

The two following tools independently consume the raw tokenizer text. They
do not load the model and do not invoke one another:

```bash
python tools/pretty_text.py --input out/output.txt --output out/output.pretty.txt
python tools/txt_to_png.py  --input out/output.txt --output out/frame.png
```

`output.ids.json` is the canonical record of emitted model row ids;
`output.txt` is their standard-tokenizer interchange form.

The cursor/pixel token protocol these tools decode is specified in
`PROTOCOL.md` in the source repo (github.com/physicsrob/torchwright_doom).
"""


def write_bundle_layout(destination: str | Path, *, prompt_text: str) -> list[Path]:
    """Copy the inference program and tools into ``destination``; write the
    executable prompt and the tools README. Returns every written path."""
    bundle = Path(destination)
    bundle.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parents[1]
    sources = {
        "infer.py": package / "infer.py",
        "tools/pretty_text.py": package / "portable" / "pretty_text.py",
        "tools/txt_to_png.py": package / "portable" / "txt_to_png.py",
    }
    written = []
    for name, source in sources.items():
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        written.append(target)
    prompt = bundle / "examples" / "e1m1_prompt.txt"
    prompt.parent.mkdir(parents=True, exist_ok=True)
    prompt.write_text(prompt_text.rstrip() + "\n", encoding="utf-8")
    tools_readme = bundle / "tools" / "README.md"
    tools_readme.write_text(_TOOLS_README, encoding="utf-8")
    return [*written, prompt, tools_readme]


def write_model_card(bundle: Path, config: RenderConfig) -> None:
    text = f"""---
library_name: transformers
pipeline_tag: text-generation
---

# TorchWright Doom — {config.map}

This is a stock Hugging Face `Phi3ForCausalLM` that renders DOOM through
ordinary autoregressive inference. The model and the data-only fast tokenizer
load through ordinary Transformers auto classes without remote code.

The bundled `examples/e1m1_prompt.txt` is the executable prompt. Run
`infer.py` (at the bundle root) to produce canonical emitted row ids and raw
tokenizer text. `tools/pretty_text.py` formats that text for reading, while
`tools/txt_to_png.py` independently decodes its cursor/pixel protocol into a
PNG — every cursor move and pixel in that protocol is a model-emitted token.
The protocol is specified in `PROTOCOL.md` in the source repo. Neither
post-processing tool participates in inference or performs geometry,
visibility, lighting, texture selection, or sorting.

**This bundle:** screen {config.screen[0]}×{config.screen[1]}, map
{config.map}, dense fp32 sharded safetensors, eager attention (the validated
implementation), greedy decode, generation bound
{config.run.max_new_tokens} new tokens.

**What running it takes:** the weights load in full fp32 (total size = the
sum of the shards; the flagship 320×200 bundle is ~98 GB, needing an
H200/B200-class GPU or multi-GPU `device_map`). The flagship render measured
~42 minutes of greedy decode on one B200 for its 53,747-token rollout from a
3,614-token prompt, scoring ~99.99% within-option color against the
reference renderer. Canonical numbers and their provenance: `FACTS.md` in
the source repo (github.com/physicsrob/torchwright_doom).
"""
    (bundle / "README.md").write_text(text, encoding="utf-8")
