# -*- coding: utf-8 -*-
"""
Exclusive OS-level file locks.

The operating system releases such a lock when the process dies, so a
crashed or killed run never leaves a stale lock behind. Locks are
non-blocking: a busy lock is reported at once, and the next scheduled run
picks the work up.
"""
import os
import sys
import threading
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Optional, Self

from llm_conductor.constants import LOCK_REGION_LENGTH, LOCK_REGION_OFFSET
from llm_conductor.exceptions import FileOperationError, LockUnavailableError


class LockStrategy(ABC):
    """Platform primitive that locks an open file descriptor."""

    @abstractmethod
    def try_lock(self, fd: int) -> bool:
        """
        Take the exclusive lock without waiting.

        Args:
            fd: Descriptor of an open file.

        Returns:
            True if the lock was taken, False if another holder has it.
        """

    @abstractmethod
    def unlock(self, fd: int) -> None:
        """
        Release a lock taken by try_lock() on the same descriptor.

        Args:
            fd: Descriptor the lock was taken on.
        """


if sys.platform == 'win32':
    import msvcrt

    class WindowsLockStrategy(LockStrategy):
        """
        Byte-range lock (LockFile) on one byte far beyond the end of the file.

        Windows byte-range locks are mandatory: locking the content would
        block reading it, even from this process. A region past the end of
        file never overlaps the data, and Windows allows locking it.
        """

        def try_lock(self, fd: int) -> bool:
            os.lseek(fd, LOCK_REGION_OFFSET, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, LOCK_REGION_LENGTH)
            except OSError:
                return False
            return True

        def unlock(self, fd: int) -> None:
            os.lseek(fd, LOCK_REGION_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, LOCK_REGION_LENGTH)

    _DEFAULT_LOCK_STRATEGY: LockStrategy = WindowsLockStrategy()

else:
    import fcntl

    class PosixLockStrategy(LockStrategy):
        """Advisory whole-file lock (flock); reading the file is never blocked."""

        def try_lock(self, fd: int) -> bool:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return True

        def unlock(self, fd: int) -> None:
            fcntl.flock(fd, fcntl.LOCK_UN)

    _DEFAULT_LOCK_STRATEGY = PosixLockStrategy()


class FileLock:
    """
    Exclusive, non-blocking lock on one file, usable as a context manager.

    Two FileLock objects on the same file conflict even inside one process
    (each opens its own descriptor), so a second run is detected the same
    way whether it is another process or not. acquire() and release() are
    serialized by an internal lock and may be called from any thread.
    """

    def __init__(self, path: str, create: bool = False,
                 strategy: Optional[LockStrategy] = None):
        """
        Args:
            path: File to lock.
            create: Create the file and its directory when missing (a
                dedicated lock file); otherwise the file must exist.
            strategy: Platform primitive; the one of the running OS by default.
        """
        self.path = path
        self._create = create
        self._strategy = strategy or _DEFAULT_LOCK_STRATEGY
        self._fd: Optional[int] = None
        self._state_lock = threading.Lock()

    @property
    def locked(self) -> bool:
        """True while this object holds the lock."""
        with self._state_lock:
            return self._fd is not None

    def acquire(self) -> None:
        """
        Take the lock without waiting.

        Raises:
            LockUnavailableError: If another holder has the lock.
            FileOperationError: If the file cannot be opened.
            RuntimeError: If this object already holds the lock.
        """
        with self._state_lock:
            if self._fd is not None:
                raise RuntimeError(f"{self.path} is already locked by this object")

            fd = self._open()
            try:
                taken = self._strategy.try_lock(fd)
            except OSError as e:
                os.close(fd)
                raise FileOperationError(f"Cannot lock {self.path}: {e}") from e

            if not taken:
                os.close(fd)
                raise LockUnavailableError(self.path)
            self._fd = fd

    def release(self) -> None:
        """Release the lock; a no-op when it is not held."""
        with self._state_lock:
            if self._fd is None:
                return
            fd, self._fd = self._fd, None
            try:
                with suppress(OSError):  # closing the descriptor releases the lock anyway
                    self._strategy.unlock(fd)
            finally:
                os.close(fd)

    def _open(self) -> int:
        """
        Open the file for locking.

        Returns:
            The descriptor.

        Raises:
            FileOperationError: If the file is missing (and create is False)
                or cannot be opened.
        """
        try:
            if self._create:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                return os.open(self.path, os.O_RDWR | os.O_CREAT)
            return os.open(self.path, os.O_RDONLY)
        except OSError as e:
            raise FileOperationError(f"Cannot open {self.path} for locking: {e}") from e

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

    def __repr__(self) -> str:
        # Reads the state without _state_lock: repr() must be safe inside acquire()/release().
        return f"FileLock({self.path!r}, locked={self._fd is not None})"
