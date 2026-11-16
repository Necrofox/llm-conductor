# -*- coding: utf-8 -*-
"""Selective prompt preview: path filter, limit, and JSONL records.

The filter decides only which eligible requests are shown. It does not decide
which leaves are processed.
"""
import json
import os
import threading
from typing import Any, Dict, FrozenSet, List, Optional

from llm_conductor.constants import (
    HINT_ORIGIN_INHERITED,
    PREVIEW_VERIFICATION_PRELIMINARY_NOTE,
)
from llm_conductor.core.io import safe_write_file
from llm_conductor.exceptions import FileOperationError
from llm_conductor.interfaces import FileOperationsProtocol, Logger
from llm_conductor.l10n.context import KeyPath, LinguisticHint


def parse_preview_path(text: str) -> str:
    """
    Parse one ``--preview-path`` value into a canonical structural identity.

    Args:
        text: JSON array of string keys and non-negative indices.

    Returns:
        ``KeyPath.report_key()`` for that array. Spacing in the input does
        not matter; a literal key ``"flat.key"`` stays one segment.

    Raises:
        ValueError: If the text is not that kind of array.
    """
    try:
        segments = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{text!r} is not a JSON array of path segments") from exc
    if not isinstance(segments, list) or any(
        not isinstance(segment, (str, int)) or isinstance(segment, bool)
        or (isinstance(segment, int) and segment < 0)
        for segment in segments
    ):
        raise ValueError(
            f"{text!r} must be a JSON array of string keys or non-negative indices"
        )
    return KeyPath(tuple(segments)).report_key()


def hint_preview_fields(hint: LinguisticHint) -> dict:
    """One final hint as shown in a preview record."""
    item: Dict[str, Any] = {
        "slot": hint.slot,
        "text": hint.instruction,
        "origin": hint.origin,
        "rule": hint.rule_name,
    }
    if hint.origin == HINT_ORIGIN_INHERITED:
        parent: List[object] = []
        if hint.parent_path:
            parent = json.loads(hint.parent_path)
        item["parent"] = parent
    return item


def build_preview_record(
        source: str,
        path: KeyPath,
        operation: str,
        hints: List[LinguisticHint],
        payload: dict,
        *,
        preliminary: bool,
) -> dict:
    """
    One JSONL preview object.

    ``path`` is the structural segment array, so a flat key and a nested path
    that share a dotted spelling stay distinct.
    """
    note = PREVIEW_VERIFICATION_PRELIMINARY_NOTE if preliminary else ""
    return {
        "source": source,
        "path": list(path.segments),
        "operation": operation,
        "hints": [hint_preview_fields(hint) for hint in hints],
        "payload": payload,
        "preliminary": preliminary,
        "note": note,
    }


def format_prompt_preview(record: dict) -> str:
    """
    Human-readable preview. The provider payload is the only pretty-printed
    JSON object, so callers can still lift it with ``index("{\\n")``.
    """
    path_text = json.dumps(record["path"], ensure_ascii=False, separators=(",", ":"))
    lines = [
        f"--- {record['operation']}: {path_text} ---",
        f"source: {record['source'] or '-'}",
        f"operation: {record['operation']}",
        f"path: {path_text}",
    ]
    hints = record["hints"]
    if not hints:
        lines.append("hints: (none)")
    else:
        lines.append("hints:")
        for hint in hints:
            parent = ""
            if "parent" in hint:
                parent_text = json.dumps(hint["parent"], ensure_ascii=False, separators=(",", ":"))
                parent = f", parent {parent_text}"
            lines.append(
                f"  {hint['slot']} [{hint['origin']} {hint['rule']}{parent}]: {hint['text']}"
            )
    if record["note"]:
        lines.append(f"note: {record['note']}")
    lines.append(json.dumps(record["payload"], ensure_ascii=False, indent=2))
    return "\n".join(lines) + "\n"


class PreviewSelection:
    """
    Shared preview budget for one run.

    ``paths`` is a set of ``KeyPath.report_key()`` values. ``None`` shows
    every eligible request. ``limit`` counts emitted requests across files
    and stages. Neither field changes processing.
    """

    def __init__(
            self,
            paths: Optional[FrozenSet[str]] = None,
            limit: Optional[int] = None,
            *,
            stdout: bool = False,
            output_path: Optional[str] = None,
    ):
        self.paths = paths
        self.limit = limit
        self.stdout = stdout
        self.output_path = output_path
        self._emitted = 0
        self._records: List[dict] = []
        self._lock = threading.Lock()

    @property
    def captures(self) -> bool:
        """True when a sink (stdout or JSONL) is active."""
        return self.stdout or self.output_path is not None

    def reserve(self, report_key: str) -> bool:
        """
        Claim one preview slot for this structural path.

        Returns:
            False when the path is filtered out or the limit is already spent.
        """
        with self._lock:
            if self.paths is not None and report_key not in self.paths:
                return False
            if self.limit is not None and self._emitted >= self.limit:
                return False
            self._emitted += 1
            return True

    def add(self, record: dict) -> None:
        """Keep one emitted record for the JSONL file."""
        with self._lock:
            self._records.append(record)

    def jsonl_text(self) -> str:
        """JSONL body, including a trailing newline when any record exists."""
        with self._lock:
            records = list(self._records)
        if not records:
            return ""
        return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)

    def write(self, file_ops: FileOperationsProtocol, logger: Optional[Logger]) -> bool:
        """
        Write the JSONL file. No output path is success.

        Returns:
            False when the write fails.
        """
        if not self.output_path:
            return True
        try:
            parent = os.path.dirname(self.output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            return safe_write_file(
                self.output_path, self.jsonl_text(), "preview JSONL", file_ops, logger,
            )
        except FileOperationError:
            return False
        except OSError as exc:
            if logger is not None:
                logger.error(f"Cannot write preview JSONL {self.output_path}: {exc}")
            return False
