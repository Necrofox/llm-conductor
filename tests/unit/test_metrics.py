# -*- coding: utf-8 -*-
"""Unit tests for MetricsCollector (llm_conductor/core/metrics.py)."""
import pytest

from llm_conductor.constants import OPERATION_TRANSLATION, OPERATION_VERIFICATION
from llm_conductor.core.metrics import MetricsCollector
from tests.support import DeadlockGuardLock


def test_successful_requests_accumulate_time_and_tokens():
    metrics = MetricsCollector()

    metrics.update(OPERATION_TRANSLATION, 0.5, failed=False, prompt_tokens=10, completion_tokens=3)
    metrics.update(OPERATION_TRANSLATION, 1.5, failed=False, prompt_tokens=20, completion_tokens=7)

    snapshot = metrics.get_metrics(OPERATION_TRANSLATION)
    assert snapshot['total_requests'] == 2
    assert snapshot['successful_requests'] == 2
    assert snapshot['failed_requests'] == 0
    assert snapshot['total_time'] == pytest.approx(2.0)
    assert (snapshot['min_time'], snapshot['max_time']) == (0.5, 1.5)
    assert (snapshot['prompt_tokens'], snapshot['completion_tokens'], snapshot['total_tokens']) == (30, 10, 40)


def test_failed_request_counts_toward_total_but_not_toward_successful():
    """total_requests is a genuine total so a run where everything failed
    still reports a non-zero request count instead of looking untouched."""
    metrics = MetricsCollector()

    metrics.update(OPERATION_VERIFICATION, 3.0, failed=True, failure_kind='network')

    snapshot = metrics.get_metrics(OPERATION_VERIFICATION)
    assert snapshot['total_requests'] == 1
    assert snapshot['successful_requests'] == 0
    assert snapshot['failed_requests'] == 1
    assert snapshot['network_failures'] == 1
    assert snapshot['parse_failures'] == 0
    assert snapshot['total_time'] == 0.0


@pytest.mark.parametrize("failure_kind, counter", [
    ('truncated', 'truncated_responses'),
    ('refusal', 'refusal_responses'),
])
def test_task62_truncated_and_refusal_are_counted_separately_from_network_and_parse(failure_kind, counter):
    metrics = MetricsCollector()

    metrics.update(OPERATION_TRANSLATION, 0.0, failed=True, failure_kind=failure_kind)

    snapshot = metrics.get_metrics(OPERATION_TRANSLATION)
    assert snapshot['failed_requests'] == 1
    assert snapshot[counter] == 1
    assert snapshot['network_failures'] == 0
    assert snapshot['parse_failures'] == 0


def test_task48_record_rejection_does_not_double_count_total_requests():
    """
    A rejected reply already went through a successful update() call (the
    transport and parsing layer saw nothing wrong) - record_rejection() must
    not add another total_requests/successful_requests, or one model call
    would be counted as two requests.
    """
    metrics = MetricsCollector()
    metrics.update(OPERATION_TRANSLATION, 0.5, failed=False, prompt_tokens=5, completion_tokens=2)

    metrics.record_rejection(OPERATION_TRANSLATION)

    snapshot = metrics.get_metrics(OPERATION_TRANSLATION)
    assert snapshot['total_requests'] == 1
    assert snapshot['successful_requests'] == 1
    assert snapshot['failed_requests'] == 0
    assert snapshot['rejected_responses'] == 1


def test_record_rejection_of_unknown_operation_is_ignored():
    metrics = MetricsCollector()

    metrics.record_rejection("summarization")

    assert metrics.get_metrics("summarization") == {}


def test_operations_are_tracked_independently():
    metrics = MetricsCollector()

    metrics.update(OPERATION_TRANSLATION, 1.0, failed=False)

    assert metrics.get_metrics(OPERATION_VERIFICATION)['total_requests'] == 0


def test_unknown_operation_is_ignored():
    metrics = MetricsCollector()

    metrics.update("summarization", 1.0, failed=False)

    assert set(metrics.get_snapshot()) == {OPERATION_TRANSLATION, OPERATION_VERIFICATION}
    assert metrics.get_metrics("summarization") == {}


def test_snapshot_is_a_copy():
    metrics = MetricsCollector()
    snapshot = metrics.get_snapshot()

    snapshot[OPERATION_TRANSLATION]['total_requests'] = 999

    assert metrics.get_metrics(OPERATION_TRANSLATION)['total_requests'] == 0


def test_task33_metrics_values_are_flat_scalars():
    """
    Snapshot shape invariant: every per-operation field is int/float.

    _snapshot_unlocked()/_metrics_unlocked() use dict.copy(); a nested
    value would share mutable state with the live collector.
    """
    metrics = MetricsCollector()
    metrics.update(OPERATION_TRANSLATION, 0.5, failed=False, prompt_tokens=1, completion_tokens=2)
    metrics.update(OPERATION_VERIFICATION, 0.0, failed=True, failure_kind='network')
    metrics.record_rejection(OPERATION_TRANSLATION)

    empty = MetricsCollector._create_empty_metrics()
    assert empty, "empty template must not be vacant"
    for key, value in empty.items():
        assert isinstance(value, (int, float)), f"empty[{key!r}] = {value!r}"

    for operation, fields in metrics.get_snapshot().items():
        assert fields, f"{operation} snapshot must not be vacant"
        for key, value in fields.items():
            assert isinstance(value, (int, float)), (
                f"{operation}.{key} = {value!r} (expected int/float)"
            )


def test_task33_snapshot_unlocked_safe_inside_locked_section():
    """Private _*_unlocked helpers must be usable while self._lock is held."""
    metrics = MetricsCollector()
    metrics._lock = DeadlockGuardLock(timeout=0.2)

    with metrics._lock:
        snapshot = metrics._snapshot_unlocked()
        one = metrics._metrics_unlocked(OPERATION_TRANSLATION)
        assert OPERATION_TRANSLATION in snapshot
        assert isinstance(one, dict)

        with pytest.raises(RuntimeError, match="deadlock"):
            metrics.get_snapshot()
