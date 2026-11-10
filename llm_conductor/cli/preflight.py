# -*- coding: utf-8 -*-
"""
Shared input preflight for ``run`` and ``run --dry-run``.

Reads every original (and draft when set), rejects root JSON ``null``,
checks draft structure/markers, and for verify-only entries without a
draft requires a readable ``translated/`` tree that matches the original
shape. Collects all faults before raising.
"""
from typing import Any, Dict, List, Optional

from llm_conductor.cli.draft_validate import (
    compare_draft_markers,
    compare_json_structure,
    format_draft_marker_errors,
    format_draft_structure_errors,
)
from llm_conductor.cli.runner import RunOptions
from llm_conductor.cli.targets import ResultLayout, RunTarget, TargetEntry
from llm_conductor.constants import STAGE_TRANSLATED
from llm_conductor.core.io import safe_read_file
from llm_conductor.exceptions import FileOperationError, TargetError
from llm_conductor.interfaces import FileOperationsProtocol, Logger


def _options_for(
        entry: TargetEntry,
        target: RunTarget,
        default_options: RunOptions,
) -> RunOptions:
    """Effective stage/strict for one entry (manifest row or CLI default)."""
    if target.is_manifest:
        return RunOptions.for_entry(entry)
    return default_options


def _reject_null_root(index: int, path: str, data: Any, label: str) -> Optional[str]:
    """Return an error block when ``data`` is JSON null; otherwise None."""
    if data is not None:
        return None
    return f"  target {index}: {label} '{path}' contains null data"


def validate_target_inputs(
        target: RunTarget,
        file_ops: FileOperationsProtocol,
        logger: Logger,
        *,
        layout: ResultLayout,
        default_options: RunOptions,
) -> None:
    """
    Ensure every input file is usable for the selected stage(s).

    Checks (all collected before raise):
    - originals are readable JSON and not root ``null``
    - drafts (when set) are readable, match original structure and markers
    - verify-only without ``draft``: ``translated/`` exists, is readable
      JSON, is not root ``null``, and matches original structure

    Args:
        target: Resolved run target.
        file_ops: File operations for reads.
        logger: Logger of the run.
        layout: Result paths (needed for verify-only ``translated/``).
        default_options: Stage/strict for a lone JSON target (CLI).

    Raises:
        TargetError: Collecting all IO / JSON / null / draft / translated
            faults, with 1-based target index when the fault is per-entry.
    """
    blocks: List[str] = []
    originals: Dict[str, Any] = {}

    for entry in target.entries:
        try:
            data = safe_read_file(
                entry.source_path, f"input data ({entry.source_path})",
                file_ops, logger, critical=True, as_text=False,
            )
        except FileOperationError as e:
            blocks.append(f"  target {entry.index}: {e}")
            continue
        null_err = _reject_null_root(
            entry.index, entry.source_path, data, "input file",
        )
        if null_err is not None:
            blocks.append(null_err)
            continue
        originals[entry.source_path] = data

    for entry in target.entries:
        if entry.draft_path is None:
            continue
        if entry.source_path not in originals:
            continue
        try:
            draft = safe_read_file(
                entry.draft_path, f"draft data ({entry.draft_path})",
                file_ops, logger, critical=True, as_text=False,
            )
        except FileOperationError as e:
            blocks.append(f"  target {entry.index}: {e}")
            continue
        null_err = _reject_null_root(
            entry.index, entry.draft_path, draft, "draft file",
        )
        if null_err is not None:
            blocks.append(null_err)
            continue
        structure_issues = compare_json_structure(originals[entry.source_path], draft)
        if structure_issues:
            blocks.append(format_draft_structure_errors(entry.index, structure_issues))
            continue
        marker_issues = compare_draft_markers(originals[entry.source_path], draft)
        if marker_issues:
            blocks.append(format_draft_marker_errors(entry.index, marker_issues))

    for entry in target.entries:
        options = _options_for(entry, target, default_options)
        if not (options.verifies and not options.translates):
            continue
        if entry.draft_path is not None:
            continue
        if entry.source_path not in originals:
            continue
        translated_path = layout.stage_path(STAGE_TRANSLATED, entry)
        try:
            translated = safe_read_file(
                translated_path, f"translated data ({translated_path})",
                file_ops, logger, critical=True, as_text=False,
            )
        except FileOperationError as e:
            blocks.append(f"  target {entry.index}: {e}")
            continue
        null_err = _reject_null_root(
            entry.index, translated_path, translated, "translated file",
        )
        if null_err is not None:
            blocks.append(null_err)
            continue
        structure_issues = compare_json_structure(
            originals[entry.source_path], translated,
        )
        if structure_issues:
            header = (
                f"  target {entry.index}: translated structure does not match "
                f"original; re-run translate before verify"
            )
            body = "\n".join(f"    {issue}" for issue in structure_issues)
            blocks.append(f"{header}\n{body}")

    if blocks:
        raise TargetError(
            f"Input validation failed for {target.path}:\n" + "\n".join(blocks)
        )
