# -*- coding: utf-8 -*-
"""
Live-tier runner: run main() against real inference and write a report.

main() runs in-process (as in the system tier) but without responses —
requests go to the host of e2e/llm.e2e.toml. The Config built inside main()
is captured by a wrapper around Config.create_from_files, so the metrics
(failed_requests, tokens) are available after the run without parsing the log.
"""
import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import pytest

from llm_conductor.constants import (
    CONDUCTOR_CONFIG_FILENAME,
    CONFIG_PATH_ENV_VAR,
    LLM_CONFIG_FILENAME,
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
    RESULT_DIR_NAME,
    TAG_MASK_PATTERN,
)
from llm_conductor.core import shutdown
from tests.system.harness import REPO_ROOT

E2E_DIR = REPO_ROOT / "e2e"
PROFILE = E2E_DIR / "llm.e2e.toml"
CONDUCTOR_PROFILE = E2E_DIR / "conductor.e2e.toml"
SAMPLE_FILE = E2E_DIR / "sample.json"
REPORTS_DIR = E2E_DIR / "reports"
# Every scenario runs the copy of sample.json with results next to it:
# <workdir>/result/<stage>/sample.json, reported as sample_<stage>.json.
RUN_TARGET = ("--target", "sample.json", "--result2target")

CYRILLIC = re.compile(r"[А-Яа-яЁё]")
# A letter that is neither Latin nor Cyrillic: CJK, Hangul, Arabic...
# (found by a live run 2026-10-08: Chinese text in a strict reply passed the
# "has Cyrillic" check because the string also contained Cyrillic).
FOREIGN_SCRIPT = re.compile(r"[^\W\d_A-Za-zА-Яа-яЁё]")
# A reply LENGTH_RATIO times longer than the source plus slack for short
# strings — suspicion of thinking out loud instead of translating
# ("Hard" -> a paragraph).
LENGTH_RATIO = 3
LENGTH_SLACK = 20
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


@dataclass
class LiveRun:
    """Result of one main() run against a real model."""
    name: str
    args: Tuple[str, ...]
    workdir: Path
    exit_code: int
    metrics: dict
    outputs: dict = field(default_factory=dict)     # stage file name -> JSON

    def failed_requests(self) -> int:
        return sum(self.metrics.get(op, {}).get('failed_requests', 0)
                   for op in (OPERATION_TRANSLATION, OPERATION_VERIFICATION))

    def rejected_responses(self) -> int:
        return sum(self.metrics.get(op, {}).get('rejected_responses', 0)
                   for op in (OPERATION_TRANSLATION, OPERATION_VERIFICATION))

    def model_requests(self) -> int:
        return sum(self.metrics.get(op, {}).get('total_requests', 0)
                   for op in (OPERATION_TRANSLATION, OPERATION_VERIFICATION))


def load_entry_module():
    """Import the package CLI module (signal/atexit/Config live there)."""
    import llm_conductor.cli.main as module
    return module


def run_scenario(module, name: str, workdir: Path, *args: str) -> LiveRun:
    """
    Run main() in workdir with the given CLI args.

    signal.signal/atexit.register are intercepted (do not touch pytest
    handlers); the global ShutdownManager and the 'conductor' logger are
    reset by hand after the run: the session fixture lives outside the
    test's autouse fixtures.
    """
    captured = {}
    original_factory = module.Config.create_from_files

    def capturing_factory(*f_args, **f_kwargs):
        config = original_factory(*f_args, **f_kwargs)
        captured['config'] = config
        return config

    exit_code = 0
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(workdir)
        mp.setenv(CONFIG_PATH_ENV_VAR, str(workdir / CONDUCTOR_CONFIG_FILENAME))
        mp.setattr(sys, "argv", ["llm-conductor.py", "run", *RUN_TARGET, *args])
        mp.setattr(module.signal, "signal", lambda *a: None)
        mp.setattr(module.atexit, "register", lambda *a: None)
        mp.setattr(module.Config, "create_from_files", staticmethod(capturing_factory))
        try:
            module.main()
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else int(exc.code is not None)
        finally:
            _reset_process_state()

    config = captured.get('config')
    metrics = config.metrics.get_snapshot() if config is not None else {}
    outputs = {f"sample_{p.parent.name}.json": json.loads(p.read_text(encoding="utf-8"))
               for p in (workdir / RESULT_DIR_NAME).glob("*/sample.json")}
    return LiveRun(name, tuple(args), workdir, exit_code, metrics, outputs)


def _reset_process_state() -> None:
    manager = shutdown._shutdown_manager
    with manager._lock:
        manager._shutdown_requested = False
        manager._shutdown_in_progress = False
        manager._config_instance = None
    logger = logging.getLogger('conductor')
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def prepare_workdir(root: Path, name: str) -> Path:
    """Run directory: the e2e profile as conductor.toml + llm.toml, and sample.json."""
    workdir = root / name
    workdir.mkdir()
    shutil.copy2(CONDUCTOR_PROFILE, workdir / CONDUCTOR_CONFIG_FILENAME)
    shutil.copy2(PROFILE, workdir / LLM_CONFIG_FILENAME)
    shutil.copy2(SAMPLE_FILE, workdir / "sample.json")
    return workdir


# ============================================================================
# Leaf walk and checks
# ============================================================================

def iter_leaves(node: Any, path: str = "") -> Iterator[Tuple[str, Any]]:
    """(path, value) for every non-container leaf in walk order."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from iter_leaves(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from iter_leaves(value, f"{path}[{index}]")
    else:
        yield path, node


def structure_violations(original: Any, result: Any, path: str = "root") -> List[str]:
    """Tree-shape mismatches: keys and their order, list lengths, non-string leaves."""
    if isinstance(original, dict):
        if not isinstance(result, dict) or list(original) != list(result):
            return [f"{path}: ключи {list(original)} -> {list(result) if isinstance(result, dict) else type(result).__name__}"]
        return [v for key in original for v in structure_violations(original[key], result[key], f"{path}.{key}")]
    if isinstance(original, list):
        if not isinstance(result, list) or len(original) != len(result):
            return [f"{path}: список длины {len(original)} -> {result!r:.60}"]
        return [v for i, (a, b) in enumerate(zip(original, result)) for v in structure_violations(a, b, f"{path}[{i}]")]
    if isinstance(original, str):
        if not isinstance(result, str):
            return [f"{path}: строка стала {type(result).__name__}"]
        if not original.strip() and result != original:
            return [f"{path}: пустая/пробельная строка изменена: {result!r}"]
        return []
    return [] if result == original and type(result) is type(original) else [f"{path}: {original!r} -> {result!r}"]


def paired_string_leaves(original: Any, result: Any) -> Iterator[Tuple[str, str, str]]:
    """(path, source, result) for non-empty string leaves."""
    result_leaves = dict(iter_leaves(result))
    for path, value in iter_leaves(original):
        if isinstance(value, str) and value.strip() and isinstance(result_leaves.get(path), str):
            yield path, value, result_leaves[path]


def tag_violations(original: Any, result: Any) -> List[str]:
    return [f"{path}: теги {sorted(TAG_MASK_PATTERN.findall(src))} -> {sorted(TAG_MASK_PATTERN.findall(out))}: {out!r}"
            for path, src, out in paired_string_leaves(original, result)
            if sorted(TAG_MASK_PATTERN.findall(src)) != sorted(TAG_MASK_PATTERN.findall(out))]


def foreign_script_violations(original: Any, result: Any) -> List[str]:
    """Third-script letters in the reply (the model 'slipped' into another language)."""
    return [f"{path}: {sorted(set(FOREIGN_SCRIPT.findall(out)))[:10]} в {out!r}"
            for path, _, out in paired_string_leaves(original, result) if FOREIGN_SCRIPT.search(out)]


def length_violations(original: Any, result: Any) -> List[str]:
    """Replies anomalously longer than the source — chatter/essay instead of a translation."""
    return [f"{path}: {len(src)} -> {len(out)} символов: {out!r:.120}"
            for path, src, out in paired_string_leaves(original, result)
            if len(out) > LENGTH_RATIO * len(src) + LENGTH_SLACK]


def foreign_latin_violations(original: Any, result: Any) -> List[str]:
    """Latin words in the reply that are not in the source string (chatter, under-translation)."""
    violations = []
    for path, src, out in paired_string_leaves(original, result):
        source_words = {w.lower() for w in LATIN_WORD.findall(src)}
        foreign = [w for w in LATIN_WORD.findall(TAG_MASK_PATTERN.sub(" ", out)) if w.lower() not in source_words]
        if foreign:
            violations.append(f"{path}: {foreign} в {out!r}")
    return violations


# ============================================================================
# Report
# ============================================================================

def write_report(runs: List[LiveRun], source: Any, model_info: str, report_dir: Path) -> Path:
    """
    Write a markdown report and copy run artefacts into report_dir.

    The path → original → translation → after-verification table is for
    human review: translation quality cannot be judged automatically.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Живой прогон {report_dir.name}", "", f"Модель: {model_info}", ""]

    lines += ["## Прогоны", "", "| Сценарий | Аргументы | Код | Запросов к модели | failed_requests |",
              "|---|---|---|---|---|"]
    for run in runs:
        lines.append(f"| {run.name} | `{' '.join(run.args)}` | {run.exit_code} | "
                     f"{run.model_requests()} | {run.failed_requests()} |")

    for run in runs:
        translated = run.outputs.get("sample_translated.json")
        final = run.outputs.get("sample_final.json")
        if final is None:
            continue
        lines += ["", f"## {run.name}", "", "| Путь | Оригинал | Перевод | Финал |", "|---|---|---|---|"]
        translated_leaves = dict(iter_leaves(translated)) if translated is not None else {}
        final_leaves = dict(iter_leaves(final))
        for path, value in iter_leaves(source):
            if isinstance(value, str) and value.strip():
                lines.append(f"| `{path}` | {_cell(value)} | {_cell(translated_leaves.get(path))} | "
                             f"{_cell(final_leaves.get(path))} |")
        shutil.copytree(run.workdir, report_dir / run.name, dirs_exist_ok=True)

    path = report_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _cell(value: Optional[Any]) -> str:
    if value is None:
        return "—"
    return str(value).replace("\n", "\\n").replace("|", "\\|")
