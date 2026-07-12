"""Structural policy gates for the production runtime architecture.

Every path-sensitive assertion is keyed on the module-level constants below,
so a cleanup move flips a constant instead of rewriting assertions
(plan_cleanup_v2.md, Workstream 1). The gates are AST-based and fast — they
must fail on drift without compiling a model.

What they enforce, lifecycle-stage by stage:

* exactly one production ``.generate()`` caller — the portable runtime file;
* the runtime imports only stdlib + torch + transformers, and no production
  module imports the runtime (it is executed, never imported);
* portable tool sources are stdlib-only and imported only by the formatter
  wrapper, staged bundle validation, and tests;
* ONNX / ONNX Runtime imports are confined to the diagnostics side;
* ``interpret/`` and ``prompt/`` never import Transformers (neither side of
  the model may load a model); ``bundle/`` never imports ``interpret/`` or
  the run orchestrator; ``prompt/`` never imports the output side or pydoom;
* the root CLI is argparse + lazy dispatch only, and importing it in a fresh
  process loads no screen-sized graph state.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "torchwright_doom"

# ---------------------------------------------------------------------------
# Path constants — each cleanup move flips one of these (plan_cleanup_v2.md).
# ---------------------------------------------------------------------------

# The sole portable inference program: the one .generate() caller, copied
# byte-identical to the bundle root and executed by production as a subprocess.
RUNTIME_PATH = "torchwright_doom/infer.py"

# Pure-stdlib standalone tool sources copied byte-for-byte into bundle tools/.
PORTABLE_SOURCES = (
    "torchwright_doom/portable/pretty_text.py",
    "torchwright_doom/portable/txt_to_png.py",
)

# The only production files allowed to import the portable sources
# (Decision 12: the formatter wrapper and staged bundle validation; tests are
# exempt by scope).
PORTABLE_IMPORTERS = (
    "torchwright_doom/interpret/formatter.py",
    "torchwright_doom/inference/hf_bundle.py",
)

# Package files allowed to import onnx / onnxruntime (diagnostic-only;
# scripts/compile_report.py is the explicit non-package exception).
ONNX_IMPORTERS = ("torchwright_doom/diagnostics/onnx.py",)

# The one production file allowed to (lazily) import diagnostics/ — the CLI's
# compile-onnx-debug dispatch path. Flips to the root cli.py in Workstream 10.
DIAGNOSTICS_IMPORTERS = ("torchwright_doom/inference/cli.py",)

# Root dispatch-only CLI; None until Workstream 10 creates it.
CLI_PATH: str | None = None

# Flips True in the Workstream 2 commit that establishes bundle layout v2
# (root infer.py + tools/) — afterwards no legacy layout name may remain.
NEW_BUNDLE_LAYOUT = True

# Flips True when torchwright_doom/inference/ is deleted (Workstream 11).
OLD_PACKAGE_REMOVED = False

_STDLIB = set(sys.stdlib_module_names) | {"__future__"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _py_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts]


def _package_files() -> list[Path]:
    return _py_files(PACKAGE)


def _production_files() -> list[Path]:
    """Non-test project sources: the package, Modal entrypoints, scripts."""
    return [
        *_package_files(),
        *sorted(ROOT.glob("modal_*.py")),
        *_py_files(ROOT / "scripts"),
    ]


def _module_of(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


@lru_cache(maxsize=None)
def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _resolved_imports(path: Path) -> frozenset[str]:
    """Fully-qualified dotted names for every import statement in the file,
    including imports nested inside functions (lazy imports count)."""
    module_parts = _module_of(path).split(".")
    package_parts = module_parts if path.name == "__init__.py" else module_parts[:-1]
    out: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package_parts[: len(package_parts) - node.level + 1]
                if node.module:
                    base_parts = [*base_parts, node.module]
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base:
                out.add(base)
            for alias in node.names:
                out.add(f"{base}.{alias.name}" if base else alias.name)
    return frozenset(out)


def _imports_module(path: Path, target: str) -> bool:
    """True if any import in ``path`` is ``target`` or lives under it."""
    return any(
        name == target or name.startswith(target + ".")
        for name in _resolved_imports(path)
    )


def _top_level_names(path: Path) -> set[str]:
    return {name.split(".", 1)[0] for name in _resolved_imports(path)}


def _has_generate_call(path: Path) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate"
        for node in ast.walk(_tree(path))
    )


def _dir_files(name: str) -> list[Path]:
    return _py_files(PACKAGE / name)


# ---------------------------------------------------------------------------
# The runtime program
# ---------------------------------------------------------------------------


def test_exactly_one_production_generate_caller() -> None:
    """AST-keyed, not a text substring: comments and docs cannot trip it."""
    callers = [path for path in _production_files() if _has_generate_call(path)]
    assert callers == [ROOT / RUNTIME_PATH]


def test_runtime_imports_only_stdlib_torch_transformers() -> None:
    names = _top_level_names(ROOT / RUNTIME_PATH)
    assert names <= _STDLIB | {"torch", "transformers"}, names - _STDLIB


def test_runtime_program_is_never_imported() -> None:
    runtime_module = _module_of(ROOT / RUNTIME_PATH)
    offenders = [
        path
        for path in _production_files()
        if path != ROOT / RUNTIME_PATH and _imports_module(path, runtime_module)
    ]
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# Portable tool sources
# ---------------------------------------------------------------------------


def test_portable_sources_are_stdlib_only() -> None:
    for rel in PORTABLE_SOURCES:
        names = _top_level_names(ROOT / rel)
        assert names <= _STDLIB, (rel, names - _STDLIB)


def test_portable_imports_are_confined() -> None:
    portable_modules = [_module_of(ROOT / rel) for rel in PORTABLE_SOURCES]
    allowed = {ROOT / rel for rel in PORTABLE_IMPORTERS}
    portable_files = {ROOT / rel for rel in PORTABLE_SOURCES}
    offenders = [
        path
        for path in _production_files()
        if path not in allowed
        and path not in portable_files
        and any(_imports_module(path, module) for module in portable_modules)
    ]
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# Diagnostics confinement
# ---------------------------------------------------------------------------


def test_onnx_imports_are_confined() -> None:
    allowed = {ROOT / rel for rel in ONNX_IMPORTERS}
    offenders = [
        path
        for path in _package_files()
        if path not in allowed
        and (_imports_module(path, "onnx") or _imports_module(path, "onnxruntime"))
    ]
    assert not offenders, offenders


def test_diagnostics_imports_are_confined_to_lazy_cli_dispatch() -> None:
    allowed = {ROOT / rel for rel in DIAGNOSTICS_IMPORTERS}
    diagnostics_dir = PACKAGE / "diagnostics"
    offenders = [
        path
        for path in _production_files()
        if path not in allowed
        and diagnostics_dir not in path.parents
        and _imports_module(path, "torchwright_doom.diagnostics")
    ]
    assert not offenders, offenders
    for rel in DIAGNOSTICS_IMPORTERS:
        for node in _tree(ROOT / rel).body:  # module top level: must be lazy
            assert all("diagnostics" not in name for name in _statement_names(node))


def _statement_names(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [node.module or "", *(alias.name for alias in node.names)]
    return []


# ---------------------------------------------------------------------------
# Lifecycle-stage dependency rules (vacuous until each package exists)
# ---------------------------------------------------------------------------


def test_model_free_sides_never_import_a_model_stack() -> None:
    """Neither the input side nor the output side may load a model."""
    for path in [*_dir_files("interpret"), *_dir_files("prompt")]:
        assert not _imports_module(path, "transformers"), path
        assert not _imports_module(path, "onnxruntime"), path
        assert not _imports_module(path, "onnx"), path
    for path in _dir_files("bundle"):
        assert not _imports_module(path, "onnxruntime"), path
        assert not _imports_module(path, "onnx"), path


def test_bundle_never_imports_interpret_or_run() -> None:
    for path in _dir_files("bundle"):
        assert not _imports_module(path, "torchwright_doom.interpret"), path
        assert not _imports_module(path, "torchwright_doom.run"), path


def test_prompt_never_imports_output_side_or_oracle() -> None:
    for path in _dir_files("prompt"):
        for forbidden in (
            "torchwright_doom.interpret",
            "torchwright_doom.bundle",
            "torchwright_doom.pydoom",
        ):
            assert not _imports_module(path, forbidden), (path, forbidden)


def test_host_package_name_does_not_exist() -> None:
    assert not (PACKAGE / "host").exists()


# ---------------------------------------------------------------------------
# Import-time safety
# ---------------------------------------------------------------------------


def test_tokenizer_package_init_performs_no_imports() -> None:
    tree = _tree(PACKAGE / "tokenizer" / "__init__.py")
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)
    )


def test_cli_is_argparse_and_lazy_dispatch_only() -> None:
    if CLI_PATH is None:
        pytest.skip("root cli.py lands in Workstream 10")
    path = ROOT / CLI_PATH
    assert not _has_generate_call(path)
    allowed_project = (
        "torchwright_doom.run",
        "torchwright_doom.bundle",
        "torchwright_doom.diagnostics",
        "torchwright_doom.config",
    )
    for name in _resolved_imports(path):
        top = name.split(".", 1)[0]
        if top in _STDLIB:
            continue
        assert any(
            name == allowed or name.startswith(allowed + ".")
            for allowed in allowed_project
        ), name


def test_importing_cli_loads_no_graph_state() -> None:
    if CLI_PATH is None:
        pytest.skip("root cli.py lands in Workstream 10")
    code = (
        "import sys\n"
        "import torchwright_doom.cli\n"
        "forbidden = ["
        "'torchwright_doom.constants', 'torchwright_doom.embedding', "
        "'torchwright_doom.asset_banks', 'torchwright_doom.tokenizer.rows', "
        "'torchwright_doom.model_graph']\n"
        "loaded = [m for m in forbidden if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


# ---------------------------------------------------------------------------
# End-state gates (flip with NEW_BUNDLE_LAYOUT / OLD_PACKAGE_REMOVED)
# ---------------------------------------------------------------------------


def test_legacy_layout_names_are_gone() -> None:
    if not NEW_BUNDLE_LAYOUT:
        pytest.skip("bundle layout v2 lands in Workstream 2")
    assert not list(PACKAGE.rglob("inference_example.py"))
    legacy = "examples/" + "infer.py"  # split so this file never matches itself
    offenders = [
        path
        for path in _production_files()
        if legacy in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders


def test_old_inference_package_is_gone() -> None:
    if not OLD_PACKAGE_REMOVED:
        pytest.skip("torchwright_doom/inference/ is deleted in Workstream 11")
    assert not (PACKAGE / "inference").exists()
    offenders = [
        path
        for path in [*_production_files(), *_py_files(ROOT / "tests")]
        if _imports_module(path, "torchwright_doom.inference")
    ]
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# Retired-runtime surfaces stay gone
# ---------------------------------------------------------------------------


def test_converter_and_custom_delivery_surfaces_are_absent() -> None:
    for path in _production_files():
        source = path.read_text(encoding="utf-8")
        for forbidden in (
            "torchwright.compiler.hf.convert",
            "convert_onnx_to_hf",
            "TorchwrightForCausalLM",
            "TorchwrightCustomForCausalLM",
            "trust_remote_code=True",
        ):
            assert forbidden not in source, (path, forbidden)


def test_no_second_generation_implementation_files() -> None:
    forbidden = {"generation.py", "hf_runtime.py", "kv_cache.py"}
    offenders = [path for path in _package_files() if path.name in forbidden]
    assert not offenders, offenders
