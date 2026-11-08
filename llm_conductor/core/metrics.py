# -*- coding: utf-8 -*-
import threading
from typing import Any, Dict, Optional

from llm_conductor.constants import (
    FAILURE_KIND_NETWORK,
    FAILURE_KIND_PARSE,
    FAILURE_KIND_REFUSAL,
    FAILURE_KIND_TRUNCATED,
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
)


class MetricsCollector:
    """
    Thread-safe collector of API performance metrics.

    Encapsulates all metrics bookkeeping; callers only see update()/get_*().

    Snapshot shape invariant: each operation's metrics dict is flat and
    contains only scalar values (int/float). _snapshot_unlocked() and
    _metrics_unlocked() rely on dict.copy(); a nested field would make
    that shallow copy share mutable state with the live collector.

    Thread-safety contract: get_snapshot() and get_metrics() each take
    self._lock and must never be called from code that already holds it -
    the lock is not reentrant. Code that needs this data while already
    holding the lock must call the private _snapshot_unlocked() /
    _metrics_unlocked() instead.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: Dict[str, Dict[str, Any]] = {
            OPERATION_TRANSLATION: self._create_empty_metrics(),
            OPERATION_VERIFICATION: self._create_empty_metrics()
        }

    @staticmethod
    def _create_empty_metrics() -> Dict[str, Any]:
        """Build an empty metrics structure for one operation type."""
        return {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'network_failures': 0,
            'parse_failures': 0,
            'truncated_responses': 0,
            'refusal_responses': 0,
            'rejected_responses': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }

    def update(self, operation_type: str, duration: float = 0.0,
               failed: bool = False, prompt_tokens: int = 0,
               completion_tokens: int = 0, failure_kind: Optional[str] = None) -> None:
        """
        Record the outcome of one API call (thread-safe).

        total_requests counts every attempt, successful or not - a run where
        every request fails still reports a non-zero total, which is what
        lets the caller print a failure summary instead of staying silent.
        A reply accepted here but later discarded by the content validator
        (llm_conductor/l10n/validation.py) is recorded through record_rejection()
        instead, since it is the same attempt, not a new one - counting it
        here too would double-count total_requests.

        Args:
            operation_type: Operation this call belongs to.
            duration: Call duration in seconds; ignored when failed is True.
            failed: Whether the call ended in a failure.
            prompt_tokens: Prompt tokens consumed, if known.
            completion_tokens: Completion tokens produced, if known.
            failure_kind: FAILURE_KIND_NETWORK, FAILURE_KIND_PARSE,
                FAILURE_KIND_TRUNCATED or FAILURE_KIND_REFUSAL when failed is
                True; ignored otherwise.
        """
        if operation_type not in self._metrics:
            return

        with self._lock:
            metrics = self._metrics[operation_type]
            metrics['total_requests'] += 1

            if failed:
                metrics['failed_requests'] += 1
                if failure_kind == FAILURE_KIND_NETWORK:
                    metrics['network_failures'] += 1
                elif failure_kind == FAILURE_KIND_PARSE:
                    metrics['parse_failures'] += 1
                elif failure_kind == FAILURE_KIND_TRUNCATED:
                    metrics['truncated_responses'] += 1
                elif failure_kind == FAILURE_KIND_REFUSAL:
                    metrics['refusal_responses'] += 1
            else:
                metrics['successful_requests'] += 1

                if duration > 0:
                    metrics['total_time'] += duration
                    metrics['min_time'] = min(metrics['min_time'], duration)
                    metrics['max_time'] = max(metrics['max_time'], duration)

            if prompt_tokens > 0:
                metrics['prompt_tokens'] += prompt_tokens
            if completion_tokens > 0:
                metrics['completion_tokens'] += completion_tokens
            if prompt_tokens > 0 or completion_tokens > 0:
                metrics['total_tokens'] += (prompt_tokens + completion_tokens)

    def record_rejection(self, operation_type: str) -> None:
        """
        Record a reply that the content validator discarded after the call.

        Does not touch total_requests/successful_requests: update() already
        counted this attempt (the transport and parsing succeeded) when it
        was made. Kept as a separate counter so print_api_metrics() can
        report it without double-counting the request total.

        Args:
            operation_type: Operation the rejected reply belongs to.
        """
        if operation_type not in self._metrics:
            return

        with self._lock:
            self._metrics[operation_type]['rejected_responses'] += 1

    def get_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a thread-safe copy of all metrics.

        Returns:
            A dict mapping each operation type to its metrics dict.
        """
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> Dict[str, Dict[str, Any]]:
        """Same as get_snapshot(), for callers that already hold self._lock."""
        return {
            op_type: metrics.copy()
            for op_type, metrics in self._metrics.items()
        }

    def get_metrics(self, operation_type: str) -> Dict[str, Any]:
        """Return a thread-safe copy of the metrics for one operation type."""
        with self._lock:
            return self._metrics_unlocked(operation_type)

    def _metrics_unlocked(self, operation_type: str) -> Dict[str, Any]:
        """Same as get_metrics(), for callers that already hold self._lock."""
        return self._metrics.get(operation_type, {}).copy()
