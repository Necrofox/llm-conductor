# -*- coding: utf-8 -*-
"""
Unit tests for ShutdownManager (llm_conductor/core/shutdown.py, doc/pipeline.md).

Each test builds its own manager instance — the global
``_shutdown_manager`` is left alone (the autouse fixture in conftest
resets it). ``signal_handler`` (sys.exit, real signals) is covered at
the system tier.
"""
from types import SimpleNamespace

from llm_conductor.constants import OPERATION_TRANSLATION, OPERATION_VERIFICATION
from llm_conductor.core.shutdown import ShutdownManager
from tests.support import FakeCacheManager, RecordingLogger


def _manager_with(cache_manager):
    manager = ShutdownManager()
    config = SimpleNamespace(cache_manager=cache_manager, logger=RecordingLogger())
    manager.register_config(config)
    return manager, config


def test_first_request_saves_both_caches_and_sets_flag():
    cache = FakeCacheManager()
    manager, _ = _manager_with(cache)

    assert manager.request_shutdown() is True

    assert manager.is_shutdown_requested()
    assert cache.saved == [OPERATION_TRANSLATION, OPERATION_VERIFICATION]


def test_repeated_request_is_rejected_without_second_save():
    cache = FakeCacheManager()
    manager, _ = _manager_with(cache)
    manager.request_shutdown()

    assert manager.request_shutdown() is False
    assert cache.saved == [OPERATION_TRANSLATION, OPERATION_VERIFICATION]


def test_forced_request_skips_saving():
    cache = FakeCacheManager()
    manager, _ = _manager_with(cache)

    assert manager.request_shutdown(force=True) is True
    assert cache.saved == []


def test_save_failure_during_shutdown_is_logged_not_raised():
    manager, config = _manager_with(FakeCacheManager(fail_on_save=True))

    assert manager.request_shutdown() is True
    assert config.logger.contains("Failed to save caches during shutdown", level='WARNING')


def test_cleanup_saves_caches_on_normal_exit():
    cache = FakeCacheManager()
    manager, _ = _manager_with(cache)

    manager.cleanup()

    assert cache.saved == [OPERATION_TRANSLATION, OPERATION_VERIFICATION]


def test_cleanup_is_skipped_after_signal_shutdown():
    """Guard against double save: atexit after a handled signal."""
    cache = FakeCacheManager()
    manager, _ = _manager_with(cache)
    manager.request_shutdown()

    manager.cleanup()

    assert cache.saved == [OPERATION_TRANSLATION, OPERATION_VERIFICATION]


def test_cleanup_logs_save_errors_without_raising():
    manager, config = _manager_with(FakeCacheManager(fail_on_save=True))

    manager.cleanup()  # must not raise at interpreter exit

    assert config.logger.contains("Failed to save caches during cleanup", level='WARNING')


def test_without_registered_config_nothing_is_saved():
    manager = ShutdownManager()

    assert manager.request_shutdown() is True
    manager.cleanup()


def test_reason_is_logged_as_a_warning_before_saving():
    """Under a scheduler nobody reads the console - the reason must reach the log file."""
    manager, config = _manager_with(FakeCacheManager())

    manager.request_shutdown(reason="SIGTERM received")

    assert config.logger.contains("SIGTERM received: run interrupted, saving caches", level='WARNING')


def test_notify_without_config_writes_to_stdout(capsys):
    ShutdownManager().notify("stopping")

    assert "stopping" in capsys.readouterr().out
