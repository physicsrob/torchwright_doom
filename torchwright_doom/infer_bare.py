from pathlib import Path

from transformers import pipeline

model_dir = Path(__file__).parent
prompt = (model_dir / "examples/e1m1_prompt.txt").read_text()
generate = pipeline("text-generation", model=str(model_dir), device_map="auto")
frame = generate(prompt, return_full_text=False)[0]["generated_text"]
Path("output.txt").write_text(frame)
