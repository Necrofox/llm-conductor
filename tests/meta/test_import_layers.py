# -*- coding: utf-8 -*-
"""
Direction of imports inside llm_conductor/.

llm_conductor/config.py is the composition root: it may import every
component, and no component may import it - not even under TYPE_CHECKING.
A component that sees the concrete Config starts depending on the whole
facade instead of a narrow protocol (AGENTS.md).
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "llm_conductor"
COMPOSITION_ROOT = SRC / "config.py"
# Process entry (former llm-conductor.py): allowed to wire Config like the
# composition root. Every other module must stay on narrow protocols.
CLI_ENTRY = SRC / "cli" / "main.py"
FORBIDDEN_MODULE = "llm_conductor.config"
_EXCLUDED = {COMPOSITION_ROOT, CLI_ENTRY}


def _imported_modules(tree: ast.AST) -> set:
    """Every module an AST imports, at any nesting level (functions, TYPE_CHECKING blocks)."""
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            # `from llm_conductor import config` imports the module as a name
            modules |= {f"{node.module}.{alias.name}" for alias in node.names}
    return modules


def _components():
    return sorted(path for path in SRC.rglob("*.py") if path not in _EXCLUDED)


@pytest.mark.parametrize("path", _components(), ids=lambda p: str(p.relative_to(SRC.parent)))
def test_component_does_not_import_the_composition_root(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    assert FORBIDDEN_MODULE not in _imported_modules(tree), (
        f"{path.relative_to(SRC.parent)} imports {FORBIDDEN_MODULE}; depend on a protocol "
        f"from llm_conductor/interfaces.py instead")


@pytest.mark.parametrize("source", [
    "import llm_conductor.config",
    "from llm_conductor.config import Config",
    "from llm_conductor import config",
    "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from llm_conductor.config import Config",
    "def f():\n    from llm_conductor.config import Config",
])
def test_detector_catches_every_import_form(source):
    assert FORBIDDEN_MODULE in _imported_modules(ast.parse(source))


def test_similarly_named_modules_are_not_mistaken_for_the_root():
    assert FORBIDDEN_MODULE not in _imported_modules(ast.parse("from llm_conductor.schema import LLM_SCHEMA"))
