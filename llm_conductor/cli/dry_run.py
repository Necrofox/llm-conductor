# -*- coding: utf-8 -*-
"""
``run --dry-run``: same resolver as a real run, no API / locks / result writes.

Prints an effective-settings table on success. Duplicate JSON keys yield one
WARNING per file (last value wins); they do not fail the plan.
"""
import json
from typing import Dict, List, Mapping, Optional, TextIO

from llm_conductor.cli.glossary import GlossarySource
from llm_conductor.cli.runner import RunOptions
from llm_conductor.cli.targets import ResultLayout, RunTarget, TargetEntry
from llm_conductor.constants import (
    EXIT_SUCCESS,
    STAGE_FINAL,
    STAGE_REDACTED,
    STAGE_TRANSLATED,
)
from llm_conductor.interfaces import ConfigProtocol, FileOperationsProtocol, Logger


def count_duplicate_json_keys(text: str) -> int:
    """
    Count repeated object keys in JSON text (last-wins semantics).

    Args:
        text: Raw JSON document.

    Returns:
        Number of duplicate key occurrences (nested objects included).
    """
    count = 0

    def hook(pairs: list) -> dict:
        nonlocal count
        seen = set()
        result = {}
        for key, value in pairs:
            if key in seen:
                count += 1
            seen.add(key)
            result[key] = value
        return result

    json.loads(text, object_pairs_hook=hook)
    return count


def scan_file_duplicate_keys(path: str, file_ops: FileOperationsProtocol) -> int:
    """
    Read a JSON file as text and count duplicate keys.

    Args:
        path: Absolute path of a JSON file.
        file_ops: File operations for the text read.

    Returns:
        Duplicate-key count; 0 when the file has none.

    Raises:
        OSError, UnicodeDecodeError, json.JSONDecodeError: On read/parse failure.
    """
    text = file_ops.read_text_with_retry(path)
    return count_duplicate_json_keys(text)


def emit_duplicate_key_warnings(
        target: RunTarget,
        file_ops: FileOperationsProtocol,
        logger: Logger,
) -> None:
    """
    Log one WARNING per original/draft file that contains duplicate keys.

    Parse failures are ignored here: shared input preflight already reported
    unreadable or invalid JSON as exit 2.
    """
    for entry in target.entries:
        for path in (entry.source_path, entry.draft_path):
            if path is None:
                continue
            try:
                n = scan_file_duplicate_keys(path, file_ops)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, TimeoutError):
                continue
            if n <= 0:
                continue
            logger.warning(
                f"target {entry.index}: {path}: {n} duplicate key(s); last value wins"
            )


def _options_for(entry: TargetEntry, target: RunTarget, default: RunOptions) -> RunOptions:
    """Effective stage/strict for one entry."""
    if target.is_manifest:
        return RunOptions.for_entry(entry)
    return default


def _llm_path_for(
        entry: TargetEntry,
        profile_paths: Mapping[str, object],
        bootstrap_llm: str,
) -> str:
    """Absolute llm.toml path shown in the plan table."""
    if entry.llm_config_path is not None:
        return entry.llm_config_path
    if bootstrap_llm in profile_paths:
        return bootstrap_llm
    return next(iter(profile_paths))


def _glossary_path_for(
        entry: TargetEntry,
        glossaries: Dict[Optional[str], GlossarySource],
) -> str:
    """Glossary path for the table, or ``-`` when empty."""
    if entry.glossary_path is not None:
        source = glossaries.get(entry.glossary_path)
        if source is not None and source.path:
            return source.path
        return entry.glossary_path
    source = glossaries.get(None, GlossarySource())
    return source.path if source.path else "-"


def _result_paths_for(
        entry: TargetEntry,
        layout: ResultLayout,
        options: RunOptions,
) -> str:
    """Absolute stage result paths that a real run would write for this entry."""
    paths: List[str] = []
    if options.translates:
        paths.append(layout.stage_path(STAGE_TRANSLATED, entry))
    if options.verifies:
        paths.append(layout.stage_path(STAGE_REDACTED, entry))
    paths.append(layout.stage_path(STAGE_FINAL, entry))
    return "; ".join(paths)


def format_effective_table(
        target: RunTarget,
        layout: ResultLayout,
        profile_paths: Mapping[str, object],
        glossaries: Dict[Optional[str], GlossarySource],
        *,
        default_options: RunOptions,
        bootstrap_llm: str,
) -> str:
    """
    Build the effective-settings table (stdout body).

    Columns: index, original, draft, stage, strict, llm_config, glossary,
    result paths. Paths are absolute. ``profile_paths`` is keyed by absolute
    llm.toml path (values unused).
    """
    headers = (
        "index", "original", "draft", "stage", "strict",
        "llm_config", "glossary", "result_paths",
    )
    rows: List[List[str]] = []
    for entry in target.entries:
        options = _options_for(entry, target, default_options)
        rows.append([
            str(entry.index),
            entry.source_path,
            entry.draft_path if entry.draft_path else "-",
            options.stage,
            "true" if options.strict else "false",
            _llm_path_for(entry, profile_paths, bootstrap_llm),
            _glossary_path_for(entry, glossaries),
            _result_paths_for(entry, layout, options),
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: List[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [fmt(list(headers)), fmt(["-" * w for w in widths])]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def print_effective_table(
        target: RunTarget,
        layout: ResultLayout,
        profile_paths: Mapping[str, object],
        glossaries: Dict[Optional[str], GlossarySource],
        *,
        default_options: RunOptions,
        bootstrap_llm: str,
        stream: Optional[TextIO] = None,
) -> None:
    """Write the effective-settings table to stdout (or ``stream``)."""
    text = format_effective_table(
        target, layout, profile_paths, glossaries,
        default_options=default_options, bootstrap_llm=bootstrap_llm,
    )
    print(text, file=stream)


def run_dry_run(
        config: ConfigProtocol,
        target: RunTarget,
        layout: ResultLayout,
        profile_paths: Mapping[str, object],
        glossaries: Dict[Optional[str], GlossarySource],
        *,
        default_options: RunOptions,
        bootstrap_llm: str,
) -> int:
    """
    Warn on duplicate keys and print the effective-settings table.

    Caller must already have run shared input preflight (``validate_target_inputs``).
    Does not call the API, acquire locks, write results, or touch the cache.

    Returns:
        ``EXIT_SUCCESS`` when the plan is clean.
    """
    emit_duplicate_key_warnings(target, config.file_ops, config.logger)
    print_effective_table(
        target, layout, profile_paths, glossaries,
        default_options=default_options, bootstrap_llm=bootstrap_llm,
    )
    config.logger.info(
        f"Dry-run OK for {target.path} ({len(target.entries)} target(s)); "
        f"no API calls, no result writes"
    )
    return EXIT_SUCCESS
