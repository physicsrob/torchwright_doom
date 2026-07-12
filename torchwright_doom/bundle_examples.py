"""Generate the self-contained examples shipped in a Doom HF bundle."""

from __future__ import annotations

import shutil
from pathlib import Path

_README = """# Reproduce the Doom render

Run stock Hugging Face inference first:

```bash
python examples/infer.py --model . --prompt examples/e1m1_prompt.txt --output out
```

The two following commands independently consume the raw tokenizer text. They
do not load the model and do not invoke one another:

```bash
python examples/pretty_text.py --input out/output.txt --output out/output.pretty.txt
python examples/txt_to_png.py --input out/output.txt --output out/frame.png
```

`output.ids.json` is the canonical record of emitted model row ids;
`output.txt` is their standard-tokenizer interchange form.
"""


def write_bundle_examples(destination: str | Path, *, prompt_text: str) -> list[Path]:
    directory = Path(destination) / "examples"
    directory.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parent
    sources = {
        "infer.py": package / "inference_example.py",
        "pretty_text.py": package / "formatter_kernel.py",
        "txt_to_png.py": package / "frame_decoder_kernel.py",
    }
    written = []
    for name, source in sources.items():
        target = directory / name
        shutil.copy2(source, target)
        written.append(target)
    prompt = directory / "e1m1_prompt.txt"
    prompt.write_text(prompt_text.rstrip() + "\n", encoding="utf-8")
    readme = directory / "README.md"
    readme.write_text(_README, encoding="utf-8")
    return [*written, prompt, readme]
