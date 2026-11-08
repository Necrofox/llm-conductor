# -*- coding: utf-8 -*-
"""Unit tests of ConsecutiveFailureTracker (llm_conductor/core/circuit_breaker.py)."""
import threading

from llm_conductor.constants import OPERATION_TRANSLATION, OPERATION_VERIFICATION
from llm_conductor.core.circuit_breaker import ConsecutiveFailureTracker


def test_trips_once_the_streak_reaches_the_threshold():
    tracker = ConsecutiveFailureTracker(max_consecutive_failures=3)

    assert tracker.record_failure(OPERATION_TRANSLATION) is False
    assert tracker.record_failure(OPERATION_TRANSLATION) is False
    assert tracker.is_tripped(OPERATION_TRANSLATION) is False

    assert tracker.record_failure(OPERATION_TRANSLATION) is True   # crosses the threshold here
    assert tracker.is_tripped(OPERATION_TRANSLATION) is True


def test_record_failure_returns_true_only_on_the_tripping_call():
    tracker = ConsecutiveFailureTracker(max_consecutive_failures=1)

    assert tracker.record_failure(OPERATION_TRANSLATION) is True
    assert tracker.record_failure(OPERATION_TRANSLATION) is False  # already tripped, silent now
    assert tracker.record_failure(OPERATION_TRANSLATION) is False


def test_success_resets_the_streak():
    tracker = ConsecutiveFailureTracker(max_consecutive_failures=2)

    tracker.record_failure(OPERATION_TRANSLATION)
    tracker.record_success(OPERATION_TRANSLATION)

    assert tracker.record_failure(OPERATION_TRANSLATION) is False
    assert tracker.is_tripped(OPERATION_TRANSLATION) is False


def test_threshold_zero_disables_the_breaker():
    tracker = ConsecutiveFailureTracker(max_consecutive_failures=0)

    for _ in range(50):
        assert tracker.record_failure(OPERATION_TRANSLATION) is False

    assert tracker.is_tripped(OPERATION_TRANSLATION) is False


def test_negative_threshold_disables_the_breaker():
    tracker = ConsecutiveFailureTracker(max_consecutive_failures=-1)

    tracker.record_failure(OPERATION_TRANSLATION)

    assert tracker.is_tripped(OPERATION_TRANSLATION) is False


def test_translation_and_verification_streaks_are_independent():
    tracker = ConsecutiveFailureTracker(max_consecutive_failures=1)

    tracker.record_failure(OPERATION_TRANSLATION)

    assert tracker.is_tripped(OPERATION_TRANSLATION) is True
    assert tracker.is_tripped(OPERATION_VERIFICATION) is False


def test_concurrent_failures_trip_exactly_once():
    """50 threads racing record_failure() against a threshold of 50 must
    report the trip exactly once, not zero or several times."""
    tracker = ConsecutiveFailureTracker(max_consecutive_failures=50)
    trips = []
    lock = threading.Lock()

    def worker():
        result = tracker.record_failure(OPERATION_TRANSLATION)
        if result:
            with lock:
                trips.append(result)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(trips) == 1
    assert tracker.is_tripped(OPERATION_TRANSLATION) is True
