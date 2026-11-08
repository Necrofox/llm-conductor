# -*- coding: utf-8 -*-
"""
System tests of graceful shutdown (doc/pipeline.md): signal_handler and
cleanup_on_exit on top of the global _shutdown_manager.

A real Ctrl+C cannot be sent to a child process on Windows:
CTRL_BREAK_EVENT arrives as SIGBREAK, and main() registers only
SIGINT/SIGTERM. The handler is therefore called directly - the logic is the
same; the end-to-end interruption of a run is in test_pipeline.py, the live
Ctrl+C stays a manual item of the §11 checklist.
"""
import signal
from types import SimpleNamespace

import pytest

from llm_conductor.constants import OPERATION_TRANSLATION, OPERATION_VERIFICATION
from llm_conductor.core import shutdown
from llm_conductor.exceptions import RunInterrupted
from tests.support import FakeCacheManager, RecordingLogger

BOTH_CACHES = [OPERATION_TRANSLATION, OPERATION_VERIFICATION]


@pytest.fixture
def registered():
    config = SimpleNamespace(cache_manager=FakeCacheManager(), logger=RecordingLogger())
    shutdown._shutdown_manager.register_config(config)
    return config


def test_first_sigint_saves_caches_and_raises_run_interrupted(registered):
    with pytest.raises(RunInterrupted) as exc:
        shutdown.signal_handler(signal.SIGINT, None)

    assert exc.value.signal_name == "SIGINT" and not exc.value.forced
    assert registered.cache_manager.saved == BOTH_CACHES
    assert shutdown.is_shutdown_requested()
    assert registered.logger.contains("SIGINT received: run interrupted, saving caches", level='WARNING')


def test_run_interrupted_is_not_an_exception():
    """The degradation policy catches Exception around model calls; an interruption must pass through."""
    assert not issubclass(RunInterrupted, Exception)


def test_second_signal_forces_exit_without_saving(registered):
    with pytest.raises(RunInterrupted):
        shutdown.signal_handler(signal.SIGINT, None)

    with pytest.raises(RunInterrupted) as exc:
        shutdown.signal_handler(signal.SIGTERM, None)

    assert exc.value.forced
    assert registered.cache_manager.saved == BOTH_CACHES          # not saved a second time
    assert registered.logger.contains("SIGTERM received again: forcing shutdown", level='WARNING')


def test_signal_without_registered_config_reports_to_stdout(capsys):
    with pytest.raises(RunInterrupted):
        shutdown.signal_handler(signal.SIGTERM, None)

    assert "SIGTERM received: run interrupted" in capsys.readouterr().out


def test_atexit_cleanup_skipped_after_signal(registered):
    with pytest.raises(RunInterrupted):
        shutdown.signal_handler(signal.SIGINT, None)

    shutdown.cleanup_on_exit()

    assert registered.cache_manager.saved == BOTH_CACHES


def test_atexit_cleanup_saves_on_normal_exit(registered):
    shutdown.cleanup_on_exit()

    assert registered.cache_manager.saved == BOTH_CACHES
