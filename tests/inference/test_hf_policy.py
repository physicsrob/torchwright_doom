from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _production_python() -> str:
    paths = [
        *sorted((ROOT / "torchwright_doom").rglob("*.py")),
        ROOT / "modal_render.py",
    ]
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_converter_and_custom_delivery_surfaces_are_absent() -> None:
    source = _production_python()
    for forbidden in (
        "torchwright.compiler.hf.convert",
        "convert_onnx_to_hf",
        "TorchwrightForCausalLM",
        "TorchwrightCustomForCausalLM",
        "trust_remote_code=True",
    ):
        assert forbidden not in source


def test_portable_inference_has_no_project_onnx_or_formatter_dependency() -> None:
    source = (ROOT / "torchwright_doom/inference_example.py").read_text()
    assert "import onnx" not in source
    assert "onnxruntime" not in source
    assert "DoomFormatter" not in source
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".", 1)[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "torchwright_doom" not in imports
    assert "torchwright" not in imports


def test_production_has_no_second_generation_implementation() -> None:
    inference = ROOT / "torchwright_doom/inference"
    forbidden = {"generation.py", "hf_runtime.py", "kv_cache.py"}
    assert forbidden.isdisjoint(path.name for path in inference.glob("*.py"))
    paths = [
        *sorted((ROOT / "torchwright_doom").rglob("*.py")),
        ROOT / "modal_render.py",
    ]
    callers = [
        path for path in paths if ".generate(" in path.read_text(encoding="utf-8")
    ]
    assert callers == [ROOT / "torchwright_doom/inference_example.py"]
