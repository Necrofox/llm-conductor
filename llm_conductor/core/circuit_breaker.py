# -*- coding: utf-8 -*-
import threading
from typing import Dict

from llm_conductor.constants import OPERATION_TRANSLATION, OPERATION_VERIFICATION


class ConsecutiveFailureTracker:
    """
    Thread-safe per-operation streak of consecutive model-call failures.

    Trips independently for translation and verification: a run-ending
    streak of one does not affect the other. A streak resets to zero on any
    success and is unaffected by cache hits or deduplication reuse, since
    neither of those calls the model.
    """

    def __init__(self, max_consecutive_failures: int):
        """
        Args:
            max_consecutive_failures: Streak length that trips the breaker.
                A value <= 0 disables it: is_tripped() always returns False.
        """
        self._threshold = max_consecutive_failures
        self._lock = threading.Lock()
        self._streaks: Dict[str, int] = {
            OPERATION_TRANSLATION: 0,
            OPERATION_VERIFICATION: 0,
        }
        self._tripped: Dict[str, bool] = {
            OPERATION_TRANSLATION: False,
            OPERATION_VERIFICATION: False,
        }

    def record_success(self, operation_type: str) -> None:
        """Reset the streak of operation_type to zero."""
        with self._lock:
            self._streaks[operation_type] = 0

    def record_failure(self, operation_type: str) -> bool:
        """
        Extend the streak of operation_type by one failure.

        Args:
            operation_type: OPERATION_TRANSLATION or OPERATION_VERIFICATION.

        Returns:
            True exactly once per operation type: on the call whose streak
            first reaches the threshold. Callers use this to log the trip
            exactly once instead of on every subsequent failure.
        """
        with self._lock:
            self._streaks[operation_type] += 1
            if (self._threshold > 0
                    and self._streaks[operation_type] >= self._threshold
                    and not self._tripped[operation_type]):
                self._tripped[operation_type] = True
                return True
            return False

    def is_tripped(self, operation_type: str) -> bool:
        """True once operation_type's streak has reached the threshold."""
        with self._lock:
            return self._tripped[operation_type]
