# -*- coding: utf-8 -*-
"""
Docs sync against the published tree.

CI must not depend on gitignored ``history/`` (backlog, HISTORY, lint debt,
audit notes). These checks cover markup integrity of the English human docs
and ``AGENTS.md``. An unclosed code fence once inverted rendering of a large
doc from mid-file to the end — keep that regression covered.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOC_DIR = REPO / "doc"
AGENTS_MD = REPO / "AGENTS.md"


def _published_markdown() -> list[Path]:
    """Markdown files that ship in the public tree."""
    paths = [AGENTS_MD] if AGENTS_MD.is_file() else []
    if DOC_DIR.is_dir():
        paths.extend(sorted(DOC_DIR.glob("*.md")))
    return paths


PUBLISHED_MD = _published_markdown()


@pytest.mark.parametrize("path", PUBLISHED_MD, ids=lambda p: p.relative_to(REPO).as_posix())
def test_markdown_code_fences_are_balanced(path: Path) -> None:
    """An unclosed ``` inverts the render of the rest of the file."""
    inside = False
    headings_in_code: list[int] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip().startswith("```"):
            inside = not inside
        elif inside and re.match(r"^#{2,4} ", line):
            headings_in_code.append(number)

    assert not inside, f"{path.name}: unclosed code fence at end of file"
    assert headings_in_code == [], (
        f"{path.name}: headings inside a code fence at lines {headings_in_code}"
    )


@pytest.mark.parametrize("path", PUBLISHED_MD, ids=lambda p: p.relative_to(REPO).as_posix())
def test_markdown_has_no_tab_separated_tables(path: Path) -> None:
    """A table pasted from a rendered view loses | and hangs on tabs."""
    tabbed = [
        n
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "\t" in line
    ]

    assert tabbed == [], f"{path.name}: tab characters instead of markdown table, lines {tabbed}"


def test_published_docs_exist() -> None:
    """Human docs and the agent index must be present in a clean checkout."""
    expected = {
        "concepts.md",
        "configuration.md",
        "cli.md",
        "prompts.md",
        "pipeline.md",
    }
    assert AGENTS_MD.is_file(), "AGENTS.md missing at repo root"
    assert DOC_DIR.is_dir(), "doc/ missing"
    found = {p.name for p in DOC_DIR.glob("*.md")}
    missing = expected - found
    assert missing == set(), f"missing published doc files: {sorted(missing)}"
