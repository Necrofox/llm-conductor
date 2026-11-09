# -*- coding: utf-8 -*-
"""
Shared input preflight for ``run`` and ``run --dry-run``.

Reads every original (and draft when set), checks JSON parseability, and
compares draft structure to the original. Collects all faults before raising.
"""
from typing import Any, Dict, List

from llm_conductor.cli.draft_validate import (
    compare_json_structure,
    format_draft_structure_errors,
)
from llm_conductor.cli.targets import RunTarget
from llm_conductor.core.io import safe_read_file
from llm_conductor.exceptions import FileOperationError, TargetError
from llm_conductor.interfaces import FileOperationsProtocol, Logger


def validate_target_inputs(
        target: RunTarget,
        file_ops: FileOperationsProtocol,
        logger: Logger,
) -> None:
    """
    Ensure every input file is readable JSON and drafts match originals.

    Args:
        target: Resolved run target.
        file_ops: File operations for reads.
        logger: Logger of the run.

    Raises:
        TargetError: Collecting all IO / JSON / draft-structure faults, with
            1-based target index when the fault is per-entry.
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
        issues = compare_json_structure(originals[entry.source_path], draft)
        if issues:
            blocks.append(format_draft_structure_errors(entry.index, issues))

    if blocks:
        raise TargetError(
            f"Input validation failed for {target.path}:\n" + "\n".join(blocks)
        )
