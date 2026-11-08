# -*- coding: utf-8 -*-
"""
Exit code of a run that may hit several problems at once.

Every problem is recorded with its code; the most severe one by
EXIT_CODE_PRIORITY becomes the process exit code.
"""
import threading
from typing import Iterable, List, NamedTuple

from llm_conductor.constants import EXIT_CODE_PRIORITY, EXIT_SUCCESS


def _checked(codes: Iterable[int]) -> List[int]:
    """Reject codes outside the contract - an unknown code has no defined severity."""
    checked = list(codes)
    unknown = [code for code in checked if code not in EXIT_CODE_PRIORITY]
    if unknown:
        raise ValueError(f"exit codes {unknown} are not part of EXIT_CODE_PRIORITY")
    return checked


def most_severe(codes: Iterable[int]) -> int:
    """
    The code that wins by EXIT_CODE_PRIORITY.

    Args:
        codes: Exit codes; each must be listed in EXIT_CODE_PRIORITY.

    Returns:
        The most severe code, EXIT_SUCCESS for no codes.

    Raises:
        ValueError: On a code missing from EXIT_CODE_PRIORITY.
    """
    ranks = [EXIT_CODE_PRIORITY.index(code) for code in _checked(codes)]
    return EXIT_CODE_PRIORITY[min(ranks)] if ranks else EXIT_SUCCESS


class ExitRecord(NamedTuple):
    """One recorded problem of a run."""

    code: int
    reason: str


class ExitStatus:
    """
    Thread-safe accumulator of the problems of one run.

    record() may be called from any thread; code and records are consistent
    snapshots taken under the same lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: List[ExitRecord] = []

    def record(self, code: int, reason: str) -> None:
        """
        Record a problem.

        Args:
            code: Its exit code from EXIT_CODE_PRIORITY.
            reason: Human-readable description for the final log line.

        Raises:
            ValueError: On a code missing from EXIT_CODE_PRIORITY.
        """
        _checked([code])
        with self._lock:
            self._records.append(ExitRecord(code, reason))

    @property
    def code(self) -> int:
        """The exit code of the run: the most severe recorded one, EXIT_SUCCESS for none."""
        with self._lock:
            return most_severe(record.code for record in self._records)

    @property
    def records(self) -> List[ExitRecord]:
        """A copy of the recorded problems, in recording order."""
        with self._lock:
            return list(self._records)
