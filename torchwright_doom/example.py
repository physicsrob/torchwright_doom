"""Minimal runnable example for this checkpoint.

This is complete executable code, not pseudocode. It loads an ordinary
Hugging Face text-generation checkpoint, runs the bundled E1M1 prompt, executes
the cursor and pixel commands emitted by the model, and writes ``frame.png``.
It imports no TorchWright or DOOM implementation.

The loop below is all of the post-processing. The transformer emits every
cursor direction, cursor coordinate, palette index, and run width. This program
only remembers the cursor, looks up RGB in DOOM's static 256-color palette, and
asks Pillow to paint those pixels. It contains no map data, geometry, visibility
tests, texture sampling, lighting calculations, or logic for choosing what gets
drawn or in what order.

This short path is intended to make the mechanism easy to inspect. For
canonical reproduction, download the checkpoint repository and use:

    python infer.py --model . --prompt examples/e1m1_prompt.txt --output out
    python tools/txt_to_png.py --input out/output.txt --output out/frame.png

That path validates the bundle, preserves the exact generated token IDs,
records termination and memory information, and validates the decoder inputs.
See ``README.md`` for details.
"""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import pipeline

MODEL = "physicsrob/torchwright-doom-e1m1-80x50"
SCREEN = (80, 50)

prompt = Path(hf_hub_download(MODEL, "examples/e1m1_prompt.txt")).read_text()
colors = json.loads(Path(hf_hub_download(MODEL, "doom_palette.json")).read_text())[
    "colors"
]

generate = pipeline(
    "text-generation", model=MODEL, device_map="auto", trust_remote_code=False
)
output = generate(prompt, return_full_text=False)[0]["generated_text"]

image = Image.new("RGB", SCREEN)
x = y = 0
advance_x = False

for token in output.split():
    command, _, arguments = token.rstrip(")").partition("(")
    if command == "setCursorDirectionX":
        advance_x = True
    elif command == "setCursorDirectionY":
        advance_x = False
    elif command == "setCursorX":
        x = int(arguments)
    elif command == "setCursorY":
        y = int(arguments)
    elif command == "pixel":
        color, width = map(int, arguments.split(","))
        image.paste(tuple(colors[color]), (x, y, x + width, y + 1))
        if advance_x:
            x += width
        else:
            y += 1

image.save("frame.png")
