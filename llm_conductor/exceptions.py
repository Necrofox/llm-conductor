# -*- coding: utf-8 -*-
"""
Exception hierarchy of the application.

Every expected failure derives from ConductorError, so the entry point can
tell it from a bug. RunInterrupted is the one exception outside that tree:
see its docstring.
"""
import builtins
from typing import Optional


class ConductorError(Exception):
    """Base class of every expected application failure."""


class APIRequestError(ConductorError):
    """A model API request failed after all retries or with a fatal status."""

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        """
        Args:
            message: Description of the failure, including the decoded response detail.
            original_exception: The underlying transport exception, if any.
        """
        super().__init__(message)
        self.original_exception = original_exception


class ConfigValidationError(ConductorError):
    """A configuration file or a configuration value is invalid."""


class FileOperationError(ConductorError):
    """A file could not be read or written."""


class FileDurabilityError(FileOperationError):
    """A file was replaced, but its directory synchronization failed.

    The destination already contains the new file. Callers must not treat
    this as a failed replacement or assume that the old file was restored.
    """


class TimeoutError(ConductorError, builtins.TimeoutError):
    """
    An operation exceeded its time limit.

    Derives from both ConductorError (caught by `except ConductorError` at
    the application boundary) and the built-in TimeoutError, which is an
    OSError - code catching the built-in TimeoutError or OSError keeps
    working unchanged.
    """


class TargetError(ConductorError):
    """The run target, its manifest or the glossary cannot be used; nothing was processed."""


class LockUnavailableError(ConductorError):
    """A file is exclusively locked by another run."""

    def __init__(self, path: str):
        """
        Args:
            path: The file whose lock is held elsewhere.
        """
        super().__init__(f"{path} is locked by another run")
        self.path = path


class RunInterrupted(BaseException):
    """
    The run was stopped by a signal (SIGINT, SIGTERM).

    Raised by the signal handler in the main thread, wherever it happens to
    be. It derives from BaseException, not ConductorError: the pipeline
    degrades on `except Exception` around every model call, and an
    interruption caught there would be swallowed instead of stopping the run.
    """

    def __init__(self, signal_name: str, forced: bool = False):
        """
        Args:
            signal_name: Name of the received signal, e.g. 'SIGINT'.
            forced: True for a repeated signal that skipped saving the caches.
        """
        super().__init__(f"{signal_name} received" + (" again, forced" if forced else ""))
        self.signal_name = signal_name
        self.forced = forced
