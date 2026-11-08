# -*- coding: utf-8 -*-
"""
Graceful shutdown on SIGINT/SIGTERM.

The first signal saves both reply caches and raises RunInterrupted in the
main thread; the entry point turns it into EXIT_INTERRUPTED. A repeated
signal raises RunInterrupted at once, without saving.
"""
import signal
import sys
import threading
from contextlib import suppress
from typing import TYPE_CHECKING, Optional

from llm_conductor.constants import OPERATION_TRANSLATION, OPERATION_VERIFICATION
from llm_conductor.exceptions import RunInterrupted

if TYPE_CHECKING:
    from llm_conductor.interfaces import ConfigProtocol


class ShutdownManager:
    """
    Thread-safe coordinator of graceful shutdown.

    Holds the Config of the running pipeline, so a signal can save its
    caches and report through its logger. The lock is an RLock because
    request_shutdown() calls other locked methods.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._shutdown_requested = False
        self._shutdown_in_progress = False
        self._config_instance: Optional['ConfigProtocol'] = None

    def register_config(self, config: Optional['ConfigProtocol']) -> None:
        """
        Register the Config whose caches a shutdown saves.

        Args:
            config: The running pipeline's Config; None unregisters it.
        """
        with self._lock:
            self._config_instance = config

    def notify(self, message: str) -> None:
        """
        Report a shutdown event: to the log of the registered Config, or to stdout without one.

        Args:
            message: Text of the report.
        """
        with self._lock:
            if self._config_instance is not None:
                self._config_instance.logger.warning(message)
                return
        # A closed or broken stdout must not turn a shutdown into a crash.
        with suppress(OSError, ValueError):
            sys.stdout.write(f"\n{message}\n")
            sys.stdout.flush()

    def request_shutdown(self, force: bool = False, reason: Optional[str] = None) -> bool:
        """
        Start the shutdown: set the flag and, unless forced, save both caches.

        Args:
            force: Skip saving the caches.
            reason: What triggered the shutdown, e.g. 'SIGINT received';
                reported at WARNING before saving.

        Returns:
            True if the shutdown was started, False if one is already in progress.
        """
        with self._lock:
            if self._shutdown_in_progress:
                return False

            self._shutdown_requested = True
            self._shutdown_in_progress = True

            if force:
                return True

            if reason:
                self.notify(f"{reason}: run interrupted, saving caches; the results are incomplete")

            if self._config_instance and self._config_instance.cache_manager:
                try:
                    self._config_instance.cache_manager.save(OPERATION_TRANSLATION)
                    self._config_instance.cache_manager.save(OPERATION_VERIFICATION)
                except Exception as e:  # noqa: BLE001 - signal path must finish and raise RunInterrupted
                    if self._config_instance:
                        self._config_instance.logger.warning(
                            f"Failed to save caches during shutdown: {e}"
                        )

            return True

    def is_shutdown_requested(self) -> bool:
        """Thread-safe read of the shutdown flag."""
        with self._lock:
            return self._shutdown_requested

    def cleanup(self) -> None:
        """Save the caches at interpreter exit, unless a signal shutdown already did."""
        with self._lock:
            if self._shutdown_requested:
                return

            if self._config_instance and self._config_instance.cache_manager:
                try:
                    self._config_instance.cache_manager.save(OPERATION_TRANSLATION)
                    self._config_instance.cache_manager.save(OPERATION_VERIFICATION)
                except Exception as e:  # noqa: BLE001 - atexit must not raise; log and swallow
                    # ThreadPool timeout wrappers can fail during interpreter
                    # finalization on Windows, so this path is a last resort.
                    if self._config_instance:
                        self._config_instance.logger.warning(
                            f"Failed to save caches during cleanup: {e}"
                        )


_shutdown_manager = ShutdownManager()


def _signal_name(signum: int) -> str:
    """Name of a signal number, e.g. 'SIGINT'; the number itself for an unknown one."""
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def signal_handler(signum, frame):
    """
    SIGINT/SIGTERM handler, registered by the entry point for the run command.

    Raises:
        RunInterrupted: Always - after saving the caches on the first signal,
            immediately on a repeated one.
    """
    signal_name = _signal_name(signum)

    if not _shutdown_manager.request_shutdown(force=False, reason=f"{signal_name} received"):
        _shutdown_manager.notify(f"{signal_name} received again: forcing shutdown without saving caches")
        raise RunInterrupted(signal_name, forced=True)

    raise RunInterrupted(signal_name)


def cleanup_on_exit():
    """atexit handler: see ShutdownManager.cleanup()."""
    _shutdown_manager.cleanup()


def is_shutdown_requested() -> bool:
    """Whether a shutdown was requested; polled by the tree traversal."""
    return _shutdown_manager.is_shutdown_requested()
