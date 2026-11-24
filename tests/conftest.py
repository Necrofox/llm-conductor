# -*- coding: utf-8 -*-
"""
Root fixtures for the test shield.

Only what every test needs regardless of tier lives here: isolation of
process-global state that the code keeps in module singletons and cannot
reset on its own.
"""
import logging
import time
from pathlib import Path

import pytest

from llm_conductor.core import shutdown

_TIER_MARKERS = ('unit', 'integration', 'system', 'live', 'meta')


def pytest_collection_modifyitems(config, items):
    """Mark each test with its tier from the path (tests/<tier>/...)."""
    for item in items:
        parts = Path(str(item.fspath)).parts
        for tier in _TIER_MARKERS:
            if tier in parts:
                item.add_marker(getattr(pytest.mark, tier))
                break


@pytest.fixture(autouse=True)
def reset_shutdown_manager():
    """
    Reset the global ShutdownManager before and after every test.

    ``_shutdown_manager`` is a module singleton in ``llm_conductor.core.shutdown``.
    A ``_shutdown_requested`` flag left by one test would make
    ``RecursiveProcessor.traverse()`` return every subsequent tree
    unprocessed — a quiet, hard-to-diagnose cross-test coupling.
    """
    def _reset():
        manager = shutdown._shutdown_manager
        with manager._lock:
            manager._shutdown_requested = False
            manager._shutdown_in_progress = False
            manager._config_instance = None

    _reset()
    yield
    _reset()


@pytest.fixture(autouse=True)
def close_conductor_log_handlers():
    """
    Close ``conductor`` logger handlers after each test.

    ``ConfigLogger.create()`` attaches a ``FileHandler`` to the
    process-wide ``logging.getLogger('conductor')``. On Windows an open
    log file blocks pytest from deleting ``tmp_path``, and leftover
    handlers leak into later tests (duplicated output into foreign files).
    """
    yield
    logger = logging.getLogger('conductor')
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


@pytest.fixture
def no_sleep(monkeypatch):
    """
    Disable ``time.sleep`` in modules that use backoff retries.

    ``make_request_with_retry`` waits ``backoff_factor ** attempt``
    seconds; ``FileOperations`` waits ``retry_delay * 2 ** attempt``.
    Without the stub, retry tests would take seconds. Both modules call
    ``time.sleep`` through the shared ``time`` module object, so that
    object itself is patched (monkeypatch restores the original).
    Returns the list of requested delays — used to assert the backoff
    sequence itself.
    """
    delays = []

    def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(time, 'sleep', fake_sleep)
    return delays
