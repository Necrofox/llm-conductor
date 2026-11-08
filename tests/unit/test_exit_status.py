# -*- coding: utf-8 -*-
"""Unit tests of the exit code contract: llm_conductor/cli/exit_status.py and EXIT_CODE_PRIORITY."""
import threading

import pytest

from llm_conductor.cli.exit_status import ExitStatus, most_severe
from llm_conductor.constants import (
    EXIT_API_UNAVAILABLE,
    EXIT_CODE_PRIORITY,
    EXIT_CONFIG_ERROR,
    EXIT_CONSECUTIVE_FAILURES,
    EXIT_FAILURE_RATIO_EXCEEDED,
    EXIT_INTERNAL_ERROR,
    EXIT_INTERRUPTED,
    EXIT_SUCCESS,
    EXIT_TARGET_LOCKED,
    EXIT_USAGE_ERROR,
)


def test_priority_is_the_agreed_order():
    """1 > 3 > 4 > 7 > 6 > 5 > 130 > 0; 2 never combines with others and sits after 3."""
    assert EXIT_CODE_PRIORITY == (1, 3, 2, 4, 7, 6, 5, 130, 0)


def test_every_exit_constant_has_a_rank():
    codes = {EXIT_SUCCESS, EXIT_INTERNAL_ERROR, EXIT_USAGE_ERROR, EXIT_CONFIG_ERROR, EXIT_API_UNAVAILABLE,
             EXIT_FAILURE_RATIO_EXCEEDED, EXIT_CONSECUTIVE_FAILURES, EXIT_TARGET_LOCKED, EXIT_INTERRUPTED}
    assert codes == set(EXIT_CODE_PRIORITY)


@pytest.mark.parametrize("codes, expected", [
    ([], EXIT_SUCCESS),
    ([EXIT_FAILURE_RATIO_EXCEEDED, EXIT_CONSECUTIVE_FAILURES], EXIT_CONSECUTIVE_FAILURES),
    ([EXIT_INTERRUPTED, EXIT_FAILURE_RATIO_EXCEEDED], EXIT_FAILURE_RATIO_EXCEEDED),
    ([EXIT_TARGET_LOCKED, EXIT_API_UNAVAILABLE], EXIT_API_UNAVAILABLE),
    ([EXIT_FAILURE_RATIO_EXCEEDED, EXIT_INTERNAL_ERROR, EXIT_CONFIG_ERROR], EXIT_INTERNAL_ERROR),
    ([EXIT_SUCCESS, EXIT_INTERRUPTED], EXIT_INTERRUPTED),
])
def test_most_severe_code_wins(codes, expected):
    assert most_severe(codes) == expected


def test_unknown_code_is_rejected():
    with pytest.raises(ValueError, match="42"):
        most_severe([EXIT_SUCCESS, 42])


def test_status_without_records_is_success():
    assert ExitStatus().code == EXIT_SUCCESS


def test_status_keeps_every_record_and_reports_the_most_severe():
    status = ExitStatus()
    status.record(EXIT_FAILURE_RATIO_EXCEEDED, "translation: 60%")
    status.record(EXIT_CONSECUTIVE_FAILURES, "breaker")

    assert status.code == EXIT_CONSECUTIVE_FAILURES
    assert [r.reason for r in status.records] == ["translation: 60%", "breaker"]


def test_status_rejects_an_unknown_code():
    status = ExitStatus()

    with pytest.raises(ValueError):
        status.record(99, "?")
    assert status.records == []


def test_status_records_from_many_threads():
    status = ExitStatus()
    threads = [threading.Thread(target=lambda: [status.record(EXIT_FAILURE_RATIO_EXCEEDED, "x")
                                                for _ in range(200)])
               for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(status.records) == 1600
