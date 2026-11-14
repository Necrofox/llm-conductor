# -*- coding: utf-8 -*-
"""
Heartbeat JSON snapshot of a run ({cache_dir}/progress_state.json).

External monitors read this file; it is a status report, not a lock and not
the source of truth for the process exit code (that remains sys.exit in main).
Write failures degrade to a WARNING so the pipeline keeps going.
"""
from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Sequence

from llm_conductor.cli.exit_status import ExitRecord
from llm_conductor.cli.glossary import GlossarySource
from llm_conductor.cli.targets import RunTarget, TargetEntry
from llm_conductor.constants import (
    HEARTBEAT_SCHEMA_VERSION,
    HEARTBEAT_STATUS_COMPLETED,
    HEARTBEAT_STATUS_INTERRUPTED,
    HEARTBEAT_STATUS_RUNNING,
    PROGRESS_STATE_FILENAME,
)
from llm_conductor.core.io import safe_write_file
from llm_conductor.exceptions import FileOperationError
from llm_conductor.interfaces import FileOperationsProtocol, Logger


class HeartbeatWriter:
    """
    Builds and rewrites the compact progress_state.json of one run.

    Implements HeartbeatProgressProtocol (begin_stage / note_leaf) for the
    processor; file and final boundaries are driven by TargetRunner.
    """

    def __init__(
        self,
        cache_dir: str,
        interval: int,
        file_ops: FileOperationsProtocol,
        logger: Logger,
        target: RunTarget,
        profile_name: str,
        stages: Sequence[str],
        glossary: GlossarySource,
    ):
        """
        Args:
            cache_dir: Resolved [cache].dir (created on first write if needed).
            interval: Leaf rewrite cadence; ``<= 0`` skips leaf-only flushes.
            file_ops: Atomic text writer.
            logger: Run logger (WARNING on a failed rewrite).
            target: Resolved run target.
            profile_name: ``[meta].name`` of the active llm.toml.
            stages: Operation types that will run, in pipeline order.
            glossary: Glossary source of the run (same facts as the INFO log).
        """
        self._path = os.path.join(cache_dir, PROGRESS_STATE_FILENAME)
        self._cache_dir = cache_dir
        self._interval = interval
        self._file_ops = file_ops
        self._logger = logger

        self._started = False
        self._final = False
        self._started_at = 0.0
        self._stage_started_at = 0.0
        self._leaves_since_flush = 0

        self._status = HEARTBEAT_STATUS_RUNNING
        self._exit_code: Optional[int] = None
        self._problems: List[Dict[str, Any]] = []

        self._target = {
            'path': target.path,
            'name': target.name,
            'kind': 'manifest' if target.is_manifest else 'file',
        }
        self._profile = {'name': profile_name}
        self._stages = list(stages)
        self._glossary = {
            'origin': glossary.origin,
            'path': glossary.path,
            'term_count': len(glossary.terms),
        }

        self._files_total = len(target.entries)
        self._all_relative = [entry.relative_path for entry in target.entries]
        self._done: List[str] = []
        self._failed: List[Dict[str, str]] = []
        self._pending: List[str] = list(self._all_relative)

        self._file_index = 0
        self._current_file: Optional[str] = None
        self._stage: Optional[str] = None
        self._leaves_done = 0
        self._leaves_total = 0

    @property
    def path(self) -> str:
        """Absolute or relative path of progress_state.json."""
        return self._path

    @property
    def started(self) -> bool:
        """True after the first successful start() call."""
        return self._started

    @property
    def finalized(self) -> bool:
        """True after finalize() has written the terminal snapshot."""
        return self._final

    def start(self) -> None:
        """Mark the run as running and write the initial snapshot."""
        if self._final:
            return
        self._started = True
        self._started_at = time.monotonic()
        self._status = HEARTBEAT_STATUS_RUNNING
        self._exit_code = None
        self._flush()

    def on_file_start(self, entry: TargetEntry, file_index: int) -> None:
        """
        Record that a file is about to be processed.

        Args:
            entry: The file being started.
            file_index: 1-based index among target entries.
        """
        if not self._started or self._final:
            return
        self._file_index = file_index
        self._current_file = entry.relative_path
        if entry.relative_path in self._pending:
            self._pending = [path for path in self._pending if path != entry.relative_path]
        self._stage = None
        self._leaves_done = 0
        self._leaves_total = 0
        self._flush()

    def on_file_done(self, relative_path: str, status_name: str) -> None:
        """
        Record the outcome of one file and rewrite the snapshot.

        Args:
            relative_path: Path under the result stages.
            status_name: FileStatus value (done / failed / crashed / aborted).
        """
        if not self._started or self._final:
            return
        if status_name == 'done':
            if relative_path not in self._done:
                self._done.append(relative_path)
        else:
            self._failed.append({'path': relative_path, 'status': status_name})
        self._current_file = None
        self._stage = None
        self._leaves_done = 0
        self._leaves_total = 0
        self._flush()

    def begin_stage(self, stage: str, leaves_total: int) -> None:
        """
        Start a translation or verification stage for the current file.

        Args:
            stage: OPERATION_TRANSLATION or OPERATION_VERIFICATION.
            leaves_total: Pre-scanned string-leaf count for this stage.
        """
        if not self._started or self._final:
            return
        self._stage = stage
        self._leaves_total = max(0, leaves_total)
        self._leaves_done = 0
        self._leaves_since_flush = 0
        self._stage_started_at = time.monotonic()
        self._flush()

    def note_leaf(self) -> None:
        """Count one finished string leaf; flush every ``interval`` leaves."""
        if not self._started or self._final:
            return
        self._leaves_done += 1
        if self._interval <= 0:
            return
        self._leaves_since_flush += 1
        if self._leaves_since_flush >= self._interval:
            self._leaves_since_flush = 0
            self._flush()

    def set_problems(self, records: Sequence[ExitRecord]) -> None:
        """Mirror ExitStatus.records into the snapshot (usually before finalize)."""
        self._problems = [{'code': record.code, 'reason': record.reason} for record in records]

    def finalize(self, status: str, exit_code: int) -> None:
        """
        Write the terminal snapshot.

        Args:
            status: HEARTBEAT_STATUS_COMPLETED or HEARTBEAT_STATUS_INTERRUPTED.
            exit_code: Same code the process will exit with.
        """
        if not self._started or self._final:
            return
        if status not in (HEARTBEAT_STATUS_COMPLETED, HEARTBEAT_STATUS_INTERRUPTED):
            status = HEARTBEAT_STATUS_COMPLETED
        self._status = status
        self._exit_code = exit_code
        self._current_file = None
        self._stage = None
        self._final = True
        self._flush()

    def _files_done_count(self) -> int:
        """Number of files that already have an outcome."""
        return len(self._done) + len(self._failed)

    def _leaf_frac(self) -> float:
        if self._leaves_total <= 0:
            return 0.0
        return min(self._leaves_done / self._leaves_total, 1.0)

    def _percent(self) -> float:
        total = self._files_total
        if total <= 0:
            return 0.0
        return 100.0 * (self._files_done_count() + self._leaf_frac()) / total

    def _speed_leaves_per_min(self) -> float:
        if self._leaves_done <= 0 or self._stage_started_at <= 0:
            return 0.0
        elapsed = time.monotonic() - self._stage_started_at
        if elapsed <= 0:
            return 0.0
        return self._leaves_done / (elapsed / 60.0)

    def _elapsed_sec(self) -> float:
        if self._started_at <= 0:
            return 0.0
        return max(0.0, time.monotonic() - self._started_at)

    def _snapshot(self) -> Dict[str, Any]:
        return {
            'schema_version': HEARTBEAT_SCHEMA_VERSION,
            'status': self._status,
            'exit_code': self._exit_code,
            'updated_at': datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'elapsed_sec': round(self._elapsed_sec(), 3),
            'target': dict(self._target),
            'profile': dict(self._profile),
            'stages': list(self._stages),
            'glossary': dict(self._glossary),
            'progress': {
                'percent': round(self._percent(), 2),
                'files_done': self._files_done_count(),
                'files_total': self._files_total,
                'file_index': self._file_index,
                'current_file': self._current_file,
                'stage': self._stage,
                'leaves_done': self._leaves_done,
                'leaves_total': self._leaves_total,
                'speed_leaves_per_min': round(self._speed_leaves_per_min(), 2),
            },
            'files': {
                'done': list(self._done),
                'failed': [dict(item) for item in self._failed],
                'pending': list(self._pending),
            },
            'problems': [dict(item) for item in self._problems],
        }

    def _flush(self) -> None:
        """Atomically rewrite progress_state.json; never raise into the pipeline."""
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            payload = json.dumps(self._snapshot(), ensure_ascii=False, separators=(',', ':'))
            wrote = safe_write_file(
                self._path, payload, 'heartbeat progress state',
                self._file_ops, self._logger,
                sync=False,
            )
            if not wrote:
                self._logger.warning(
                    f"Heartbeat write to {self._path} failed; continuing the run"
                )
        except FileOperationError as e:
            self._logger.warning(
                f"Heartbeat write to {self._path} failed ({e}); continuing the run"
            )
        except OSError as e:
            self._logger.warning(
                f"Heartbeat write to {self._path} failed ({e}); continuing the run"
            )
