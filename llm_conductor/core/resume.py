# -*- coding: utf-8 -*-
"""
Stage-local resume index (Issue #4 MVP).

Orthogonal to the LLM response cache: values are mirrored into the normal
stage tree (``translated/`` / ``redacted/``); this module tracks which leaves
are done via the full ``cache_key``, and rebuilds a partial stage on flush /
SIGINT. Explicit ``run --continue`` is required to reuse an index.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from llm_conductor.constants import (
    CACHE_KEY_VERSION,
    DEFAULT_RESUME_FLUSH_INTERVAL,
    RESUME_SCHEMA_VERSION,
)
from llm_conductor.core.io import safe_read_file, safe_write_file
from llm_conductor.exceptions import FileOperationError
from llm_conductor.interfaces import FileOperationsProtocol, Logger


@dataclass(frozen=True)
class ResumeLeafEntry:
    """One accepted leaf stored for continue / partial flush."""

    cache_key: str
    value: str


@dataclass(frozen=True)
class ResumeIndex:
    """On-disk resume document for one file and one stage."""

    schema_version: int
    stage: str
    operation: str
    run_fingerprint: str
    relative_path: str
    leaves: Dict[str, ResumeLeafEntry]


def source_fingerprint(source: Any) -> str:
    """Stable digest of the input tree (structure + leaf values)."""
    payload = json.dumps(source, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.md5(payload.encode('utf-8')).hexdigest()


def build_run_fingerprint(
    *,
    operation_type: str,
    model: str,
    system_prompt: str,
    prompt_template: str,
    mode: str,
    sampling_identity: str,
    api_url: str,
    allowed_scripts: Tuple[str, ...],
    content_identity: Tuple[str, str],
    source_digest: str,
) -> str:
    """
    File-level identity for hard-fail on ``--continue``.

    Path-specific hints/glossary stay in the per-leaf ``cache_key``; this
    fingerprint catches profile / prompt / source churn for the whole file.
    """
    payload = [
        CACHE_KEY_VERSION,
        operation_type,
        model,
        system_prompt,
        prompt_template,
        mode,
        sampling_identity,
        api_url,
        sorted(set(allowed_scripts)),
        list(content_identity),
        source_digest,
    ]
    raw = json.dumps(payload, ensure_ascii=False)
    return hashlib.md5(raw.encode('utf-8')).hexdigest()


def overlay_resume_values(source: Any, values: Mapping[str, str]) -> Any:
    """
    Deep-copy ``source`` and set string leaves from ``report_key`` → value.

    Unknown or structurally incompatible paths are skipped (fingerprint
    mismatch should already have refused continue).
    """
    tree = copy.deepcopy(source)
    for report_key, value in values.items():
        try:
            segments = json.loads(report_key)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(segments, list):
            continue
        _set_at_path(tree, segments, value)
    return tree


def _set_at_path(tree: Any, segments: list, value: str) -> bool:
    """Set a leaf; return False when the path does not exist or types disagree."""
    if not segments:
        return False
    node = tree
    for segment in segments[:-1]:
        if (
            (isinstance(node, dict) and isinstance(segment, str) and segment in node)
            or (
                isinstance(node, list)
                and isinstance(segment, int)
                and 0 <= segment < len(node)
            )
        ):
            node = node[segment]
        else:
            return False
    last = segments[-1]
    if isinstance(node, dict) and isinstance(last, str) and last in node:
        node[last] = value
        return True
    if isinstance(node, list) and isinstance(last, int) and 0 <= last < len(node):
        node[last] = value
        return True
    return False


def load_resume_index(
    path: str,
    file_ops: FileOperationsProtocol,
    logger: Optional[Logger] = None,
) -> Optional[ResumeIndex]:
    """
    Load a resume index from disk.

    Returns:
        The index, or None when the file is missing.

    Raises:
        FileOperationError: If the file exists but is unreadable or corrupt.
    """
    data = safe_read_file(path, "resume index", file_ops, logger)
    if data is None:
        return None
    if not isinstance(data, dict):
        raise FileOperationError(f"resume index {path} is not a JSON object")
    version = data.get('schema_version')
    if version != RESUME_SCHEMA_VERSION:
        raise FileOperationError(
            f"resume index {path} has unsupported schema_version {version!r} "
            f"(expected {RESUME_SCHEMA_VERSION}); delete it or re-run without --continue"
        )
    stage = data.get('stage')
    operation = data.get('operation')
    run_fingerprint = data.get('run_fingerprint')
    relative_path = data.get('relative_path')
    leaves_raw = data.get('leaves')
    if not isinstance(stage, str) or not stage:
        raise FileOperationError(f"resume index {path}: missing stage")
    if not isinstance(operation, str) or not operation:
        raise FileOperationError(f"resume index {path}: missing operation")
    if not isinstance(run_fingerprint, str) or not run_fingerprint:
        raise FileOperationError(f"resume index {path}: missing run_fingerprint")
    if not isinstance(relative_path, str):
        raise FileOperationError(f"resume index {path}: missing relative_path")
    if not isinstance(leaves_raw, dict):
        raise FileOperationError(f"resume index {path}: leaves must be an object")

    leaves: Dict[str, ResumeLeafEntry] = {}
    for leaf_path, entry_raw in leaves_raw.items():
        if not isinstance(leaf_path, str) or not isinstance(entry_raw, dict):
            continue
        cache_key = entry_raw.get('cache_key')
        value = entry_raw.get('value')
        if not isinstance(cache_key, str) or not isinstance(value, str):
            continue
        leaves[leaf_path] = ResumeLeafEntry(cache_key=cache_key, value=value)

    return ResumeIndex(
        schema_version=RESUME_SCHEMA_VERSION,
        stage=stage,
        operation=operation,
        run_fingerprint=run_fingerprint,
        relative_path=relative_path,
        leaves=leaves,
    )


def resume_index_to_dict(index: ResumeIndex) -> Dict[str, Any]:
    """JSON document written under ``resume/<relative>``."""
    return {
        'schema_version': index.schema_version,
        'stage': index.stage,
        'operation': index.operation,
        'run_fingerprint': index.run_fingerprint,
        'relative_path': index.relative_path,
        'leaves': {
            path: {'cache_key': entry.cache_key, 'value': entry.value}
            for path, entry in index.leaves.items()
        },
    }


class StageResumeSession:
    """
    In-memory journal of accepted leaves for one stage walk.

    Survives ``RunInterrupted`` unwinding of ``traverse`` so SIGINT can still
    flush a partial stage tree + index.
    """

    def __init__(
        self,
        *,
        source: Any,
        stage: str,
        operation: str,
        relative_path: str,
        run_fingerprint: str,
        stage_path: str,
        index_path: str,
        file_ops: FileOperationsProtocol,
        logger: Logger,
        continue_mode: bool = False,
        prior: Optional[ResumeIndex] = None,
        flush_interval: int = DEFAULT_RESUME_FLUSH_INTERVAL,
        on_flush: Optional[Callable[[], None]] = None,
    ) -> None:
        self.source = source
        self.stage = stage
        self.operation = operation
        self.relative_path = relative_path
        self.run_fingerprint = run_fingerprint
        self.stage_path = stage_path
        self.index_path = index_path
        self.file_ops = file_ops
        self.logger = logger
        self.continue_mode = continue_mode
        self.flush_interval = max(1, flush_interval)
        self._on_flush = on_flush
        self._lock = threading.Lock()
        self._completed: Dict[str, ResumeLeafEntry] = {}
        self._since_flush = 0
        self._prior: Dict[str, ResumeLeafEntry] = {}
        if continue_mode and prior is not None:
            self._prior = dict(prior.leaves)

    def lookup(self, leaf_path: str, cache_key: str) -> Optional[str]:
        """Return a prior value when ``--continue`` and ``cache_key`` still match."""
        if not self.continue_mode:
            return None
        entry = self._prior.get(leaf_path)
        if entry is None or entry.cache_key != cache_key:
            return None
        return entry.value

    def note_accepted(self, leaf_path: str, cache_key: str, value: str) -> None:
        """Record an accepted leaf; may trigger a periodic flush."""
        should_flush = False
        with self._lock:
            self._completed[leaf_path] = ResumeLeafEntry(cache_key=cache_key, value=value)
            self._since_flush += 1
            if self._since_flush >= self.flush_interval:
                self._since_flush = 0
                should_flush = True
        if should_flush:
            self.flush()

    def snapshot_values(self) -> Dict[str, str]:
        """Copy of accepted leaf values (report_key → string)."""
        with self._lock:
            return {path: entry.value for path, entry in self._completed.items()}

    def snapshot_index(self) -> ResumeIndex:
        """Copy of the in-memory index document."""
        with self._lock:
            leaves = dict(self._completed)
        return ResumeIndex(
            schema_version=RESUME_SCHEMA_VERSION,
            stage=self.stage,
            operation=self.operation,
            run_fingerprint=self.run_fingerprint,
            relative_path=self.relative_path,
            leaves=leaves,
        )

    def flush(self, *, stage_tree: Optional[Any] = None) -> bool:
        """
        Write stage tree + resume index.

        Args:
            stage_tree: When set, write this tree as the stage file (successful
                end of traverse). When None, rebuild from ``source`` + completed
                leaves (SIGINT / periodic flush).

        Returns:
            True when both writes succeed.
        """
        index = self.snapshot_index()
        if stage_tree is not None:
            tree = stage_tree
        else:
            values = {path: entry.value for path, entry in index.leaves.items()}
            tree = overlay_resume_values(self.source, values)
        stage_ok = safe_write_file(
            self.stage_path, tree, f"{self.stage} output (resume flush)",
            self.file_ops, self.logger,
        )
        index_ok = safe_write_file(
            self.index_path, resume_index_to_dict(index),
            f"resume index for {self.relative_path}",
            self.file_ops, self.logger,
        )
        if stage_ok and index_ok and self._on_flush is not None:
            self._on_flush()
        return bool(stage_ok and index_ok)

    def flush_for_shutdown(self) -> None:
        """Signal-safe flush: log failures, never raise."""
        try:
            if not self.flush():
                self.logger.warning(
                    f"Resume flush during shutdown did not fully succeed "
                    f"({self.stage_path})"
                )
        except Exception as exc:  # noqa: BLE001 - signal path must finish
            self.logger.warning(f"Failed to flush stage resume during shutdown: {exc}")
