# -*- coding: utf-8 -*-
"""
Guard: product code must not cite tracker numbers.

Docstrings and comments in llm_conductor/ and the entry point explain
behaviour, not which backlog item introduced it (AGENTS.md). Tracker
refs stay allowed in tests/doc (xfail reason=, HISTORY). This check only
walks the runtime surface so those refs cannot grow back into shipped
modules.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "llm_conductor"
ENTRY = REPO / "llm-conductor.py"
TASK_REF_RE = re.compile(r"Task #")


def _python_surfaces():
    paths = sorted(PACKAGE.rglob("*.py"))
    paths.append(ENTRY)
    return paths


def test_package_and_entry_have_no_task_number_refs():
    offenders = []
    for path in _python_surfaces():
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if TASK_REF_RE.search(line):
                rel = path.relative_to(REPO)
                offenders.append(f"{rel}:{number}: {line.strip()}")

    assert offenders == [], (
        "Task # must not appear in llm_conductor/ or llm-conductor.py "
        f"({len(offenders)} hit(s)):\n" + "\n".join(offenders)
    )
