# -*- coding: utf-8 -*-
import json
import os
import sys
import tempfile
import time
from typing import Callable, Optional, TextIO, Union

from llm_conductor.constants import (
    BYTES_PER_MIB,
    DEFAULT_FILE_OPERATION_TIMEOUT,
    DEFAULT_LARGE_FILE_THRESHOLD_MB,
    JSON_FILE_INDENT,
    RETRYABLE_IO_ERRNOS,
)
from llm_conductor.exceptions import FileDurabilityError, FileOperationError
from llm_conductor.interfaces import FileOperationsProtocol, Logger
from llm_conductor.utils import run_with_timeout


class FileOperations:
    """
    File reads and writes behind one interface: per-operation timeouts,
    retries and atomic replacement of the target file.

    Holds no mutable state after construction, so one instance is shared
    by every component of a run.
    """

    def __init__(self, logger: Optional[Logger] = None, timeout: int = DEFAULT_FILE_OPERATION_TIMEOUT,
                 large_file_threshold_mb: int = DEFAULT_LARGE_FILE_THRESHOLD_MB):
        """
        Args:
            logger: Optional logger for diagnostics.
            timeout: Timeout of a single read or write, seconds.
            large_file_threshold_mb: Size above which safe_write_file() warns.
        """
        self.logger = logger
        self.timeout = timeout
        self._large_file_threshold_mb = large_file_threshold_mb

    @property
    def large_file_threshold_mb(self) -> int:
        """Size in MB above which a write is reported as a large file."""
        return self._large_file_threshold_mb

    @staticmethod
    def _is_retryable_io_error(exc: BaseException) -> bool:
        """
        True when a file-system error is worth retrying with backoff.

        PermissionError (file locked by another process on Windows) and
        TimeoutError are always transient. Other OSError values retry only
        for errno in RETRYABLE_IO_ERRNOS. Permanent path errors
        (FileNotFoundError, NotADirectoryError, IsADirectoryError) never
        retry - a second attempt cannot create a missing directory.
        """
        if isinstance(exc, (PermissionError, TimeoutError)):
            return True
        if isinstance(exc, (FileNotFoundError, NotADirectoryError, IsADirectoryError)):
            return False
        if isinstance(exc, OSError):
            return getattr(exc, 'errno', None) in RETRYABLE_IO_ERRNOS
        return False

    def read_json(self, file_path: str) -> dict:
        """
        Read a JSON file with a timeout.

        Args:
            file_path: Path to the file.

        Returns:
            Parsed JSON data.

        Raises:
            TimeoutError: If the read exceeds the timeout.
            FileNotFoundError: If the file is missing.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        def _read() -> dict:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        return run_with_timeout(_read, self.timeout, f"read_json({file_path})")

    def read_text_with_retry(self, file_path: str, encoding: str = 'utf-8',
                              max_retries: int = 3, retry_delay: float = 1.0) -> str:
        """
        Read a text file (e.g. TOML) with timeout and retry.

        Unlike read_json, retries with exponential backoff on transient
        filesystem errors (file briefly locked, network volume unavailable).

        Args:
            file_path: Path to the file.
            encoding: File encoding.
            max_retries: Maximum attempts.
            retry_delay: Base delay between attempts.

        Returns:
            File contents.

        Raises:
            TimeoutError: If the last attempt times out.
            FileNotFoundError: If the file is missing.
            OSError: For other filesystem errors on the last attempt.
        """
        last_error: BaseException = OSError(f"read_text_with_retry: max_retries must be >= 1, got {max_retries}")

        def _read() -> str:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()

        for attempt in range(max_retries):
            try:
                return run_with_timeout(_read, self.timeout, f"read_text({file_path})")
            except OSError as e:
                if not self._is_retryable_io_error(e):
                    raise
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))

        raise last_error

    @staticmethod
    def _sync_parent_directory(directory: str) -> None:
        """Persist the replacement's directory entry on Linux."""
        if sys.platform != 'linux':
            return
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _write_atomic(self, file_path: str, encoding: str, writer: Callable[[TextIO], object], *,
                      sync: bool = True, on_written: Optional[Callable[[int], None]] = None) -> None:
        """
        Shared atomic write via a temporary file.

        Creates a temp file in the target directory, lets ``writer`` fill it,
        flushes and closes it, then replaces the destination. With sync=True,
        fsyncs the temporary file first and its parent directory after the
        replacement on Linux. Other platforms only synchronize file contents.

        Args:
            file_path: Destination path.
            encoding: File encoding.
            writer: Callback that writes into an open text file.
            sync: Synchronize file contents and, on Linux, the directory entry.
            on_written: Optional callback receiving the actual serialized byte
                size after a successful write. May run in the IO worker thread.
                Diagnostic callback failures are logged without failing the write.

        Raises:
            TimeoutError: If the write exceeds the timeout.
            OSError: On filesystem failures.
            FileDurabilityError: The destination was replaced, but directory
                synchronization failed; its durability is not confirmed.
        """
        def _do_write() -> None:
            # Same directory as the target so rename stays atomic on the volume.
            dir_name = os.path.dirname(file_path) or '.'
            base_name = os.path.basename(file_path)

            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.tmp',
                prefix=f'.{base_name}.',
                dir=dir_name,
                text=True
            )

            try:
                with os.fdopen(temp_fd, 'w', encoding=encoding) as tmp_file:
                    writer(tmp_file)
                    tmp_file.flush()
                    written_size = os.fstat(tmp_file.fileno()).st_size
                    if sync:
                        os.fsync(tmp_file.fileno())

                os.replace(temp_path, file_path)
                if sync:
                    try:
                        self._sync_parent_directory(dir_name)
                    except OSError as exc:
                        raise FileDurabilityError(
                            f"Replaced {file_path}, but parent directory synchronization failed; "
                            "durability is not confirmed"
                        ) from exc
                if on_written is not None:
                    try:
                        on_written(written_size)
                    except Exception as exc:  # noqa: BLE001 - diagnostics must not retry a committed write
                        self._log_warning(f"Size diagnostics failed for {file_path}: {exc}")

            except Exception:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                raise

        run_with_timeout(_do_write, self.timeout, f"write({file_path})")

    def write_json_atomic(self, file_path: str, data: dict, encoding: str = 'utf-8', *,
                          sync: bool = True, on_written: Optional[Callable[[int], None]] = None) -> None:
        """
        Atomically write a JSON file via a temporary file.

        Args:
            file_path: Destination path.
            data: JSON-serializable data.
            encoding: File encoding.
            sync: Synchronize contents and the Linux parent directory entry.
            on_written: Optional byte-size callback after success (IO thread).

        Raises:
            TimeoutError: If the write exceeds the timeout.
            OSError: On filesystem failures.
            FileDurabilityError: Replacement succeeded but directory sync failed.
        """
        self._write_atomic(
            file_path, encoding, lambda f: json.dump(data, f, ensure_ascii=False, indent=JSON_FILE_INDENT),
            sync=sync, on_written=on_written,
        )

    def write_text_atomic(self, file_path: str, data: str, encoding: str = 'utf-8', *,
                          sync: bool = True, on_written: Optional[Callable[[int], None]] = None) -> None:
        """
        Atomically write a text file (e.g. TOML) via a temporary file.

        Args:
            file_path: Destination path.
            data: File contents.
            encoding: File encoding.
            sync: Synchronize contents and the Linux parent directory entry.
            on_written: Optional byte-size callback after success (IO thread).

        Raises:
            TimeoutError: If the write exceeds the timeout.
            OSError: On filesystem failures.
            FileDurabilityError: Replacement succeeded but directory sync failed.
        """
        self._write_atomic(file_path, encoding, lambda f: f.write(data), sync=sync, on_written=on_written)

    def _write_with_retry(self, file_path: str, encodings: list[str], max_retries: int, retry_delay: float,
                           atomic_write: Callable[[str, str], None]) -> bool:
        """
        Shared retry with exponential backoff and an encoding fallback list.

        Used by write_json_with_retry and write_text_with_retry; the actual
        atomic write is delegated to ``atomic_write(file_path, encoding)``.

        Args:
            file_path: Destination path.
            encodings: Codecs to try in order.
            max_retries: Max attempts per encoding.
            retry_delay: Base delay between attempts.
            atomic_write: One-shot atomic write callback.

        Returns:
            True on success; False if every encoding/attempt failed.
        """
        for encoding in encodings:
            for attempt in range(max_retries):
                try:
                    atomic_write(file_path, encoding)
                    return True

                except UnicodeEncodeError:
                    # Encoding mismatch - try the next codec, no backoff.
                    break

                except OSError as e:
                    if not self._is_retryable_io_error(e):
                        return False
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (2 ** attempt))
                    else:
                        # Exhausted retries for this path error; another
                        # encoding cannot fix a missing directory or lock.
                        return False

        return False

    def write_json_with_retry(self, file_path: str, data: dict,
                               max_retries: int = 3, retry_delay: float = 1.0, *,
                               sync: bool = True, on_written: Optional[Callable[[int], None]] = None) -> bool:
        """
        Write JSON with retry and fallback encodings.

        Args:
            file_path: Destination path.
            data: JSON-serializable data.
            max_retries: Max attempts per encoding.
            retry_delay: Base delay between attempts.
            sync: Synchronize contents and the Linux parent directory entry.
            on_written: Optional byte-size callback after success (IO thread).

        Returns:
            True on success; False on total failure.

        Raises:
            FileDurabilityError: Replacement succeeded but directory sync failed.
                This is not retried as a failed replacement.
        """
        encodings = ['utf-8', 'utf-8-sig', 'cp1251', 'latin-1']
        return self._write_with_retry(
            file_path, encodings, max_retries, retry_delay,
            lambda path, encoding: self.write_json_atomic(
                path, data, encoding, sync=sync, on_written=on_written)
        )

    def write_text_with_retry(self, file_path: str, data: str,
                               max_retries: int = 3, retry_delay: float = 1.0, *,
                               sync: bool = True, on_written: Optional[Callable[[int], None]] = None) -> bool:
        """
        Write a text file (e.g. TOML) with retry.

        Unlike write_json_with_retry this does not walk encoding fallbacks —
        text config formats (TOML) are UTF-8 by specification.

        Args:
            file_path: Destination path.
            data: File contents.
            max_retries: Max attempts.
            retry_delay: Base delay between attempts.
            sync: Synchronize contents and the Linux parent directory entry.
            on_written: Optional byte-size callback after success (IO thread).

        Returns:
            True on success; False on total failure.

        Raises:
            FileDurabilityError: Replacement succeeded but directory sync failed.
                This is not retried as a failed replacement.
        """
        return self._write_with_retry(
            file_path, ['utf-8'], max_retries, retry_delay,
            lambda path, encoding: self.write_text_atomic(
                path, data, encoding, sync=sync, on_written=on_written)
        )

    def _log_debug(self, message: str):
        """Log at DEBUG if a logger is attached."""
        if self.logger:
            self.logger.debug(message)

    def _log_warning(self, message: str):
        """Log at WARNING if a logger is attached."""
        if self.logger:
            self.logger.warning(message)

    def _log_error(self, message: str, include_traceback: bool = False):
        """Log at ERROR if a logger is attached."""
        if self.logger:
            self.logger.error(message, include_traceback)

def safe_read_file(file_path: str, description: str, file_ops: FileOperationsProtocol,
                    logger: Optional[Logger] = None, critical: bool = True,
                    as_text: Optional[bool] = None) -> Union[dict, str, None]:
    """
    Read a file as JSON or as raw text, turning every failure into one outcome.

    Args:
        file_path: Path to the file.
        description: Human-readable name of the file for messages.
        file_ops: File operations used for the read.
        logger: Logger for messages.
        critical: Raise on failure instead of returning None.
        as_text: Read mode. None picks it by extension (.json/.cache -> dict,
            anything else, including .toml -> text); True/False forces it.

    Returns:
        A dict (JSON mode), a str (text mode), or None when the read failed
        and critical is False.

    Raises:
        FileOperationError: On a read failure when critical is True.
    """
    if as_text is None:
        ext = os.path.splitext(file_path)[1].lower()
        as_text = ext not in ('.json', '.cache')

    def log_msg(msg: str, is_error: bool = False):
        """Log through the optional logger."""
        if logger:
            if is_error:
                logger.error(msg)
            else:
                logger.warning(msg)

    def fail(reason: str, source: Optional[BaseException]) -> None:
        """Single failure path: raise when critical, otherwise warn and fall through to None."""
        if critical:
            log_msg(f"Error: {reason}. Exiting.", is_error=True)
            raise FileOperationError(f"Failed to read {description}: {reason}") from source
        log_msg(f"Warning: {reason}. Proceeding without {description}.")

    try:
        if as_text:
            return file_ops.read_text_with_retry(file_path)
        return file_ops.read_json(file_path)
    except FileNotFoundError as e:
        fail(f"{description} not found", e)
        return None
    except TimeoutError as e:
        fail(f"timeout reading {description}", e)
        return None
    except json.JSONDecodeError as e:
        fail(f"{description} contains invalid JSON. Error: {e}", e)
        return None
    except PermissionError as e:
        fail(f"no permission to read {description}", e)
        return None
    except UnicodeDecodeError as e:
        fail(f"decoding error in {description}. Check encoding. Error: {e}", e)
        return None
    except Exception as e:  # noqa: BLE001 - unexpected read errors fold into fail()/None
        error_type = type(e).__name__

        if logger:
            is_debug = hasattr(logger, 'is_debug_mode') and logger.is_debug_mode
            if is_debug or critical:
                logger.error(f"Exception in safe_read_file: {error_type}: {e}", include_traceback=is_debug)

            if isinstance(e, AttributeError) and 'SIGALRM' in str(e):
                logger.error(
                    "SIGALRM not available (Windows detected). "
                    "File timeout features disabled. This is a code bug - please report."
                )

            if critical:
                logger.error("About to raise FileOperationError from safe_read_file")
                logger.error(f"  File: {file_path}")
                logger.error(f"  Description: {description}")
                logger.error(f"  Exception type: {error_type}")
                logger.error(f"  Exception message: {e}")

        fail(f"unexpected error reading {description} ({error_type}: {e})", e)
        return None

def safe_write_file(file_path: str, data: Union[dict, str], description: str,
                    file_ops: FileOperationsProtocol, logger: Optional[Logger] = None,
                    max_retries: int = 3, retry_delay: float = 1.0, *, sync: bool = True) -> bool:
    """
    Write a file atomically with retries and an emergency fallback copy.

    The mode follows the type of data: dict -> JSON, str -> text (e.g. TOML).

    Args:
        file_path: Path to the file.
        data: Content (dict for JSON, str for text).
        description: Human-readable name of the file for messages.
        file_ops: File operations used for the write; also supplies the
            large-file warning threshold.
        logger: Logger for messages.
        max_retries: Attempts before falling back.
        retry_delay: Base delay between attempts, seconds.
        sync: Synchronize contents and the Linux parent directory entry. False
            keeps atomic replacement without requesting durable storage.

    Returns:
        True on success, False on a write error that was only logged.

    Raises:
        FileOperationError: When the regular write failed, whether or not the
            fallback copy {file_path}.fallback_<ts>.{json|txt} was saved.
        FileDurabilityError: A file was replaced but directory sync failed.
            No replacement retry or emergency fallback is attempted in this case.
    """
    is_debug = hasattr(logger, 'is_debug_mode') and logger.is_debug_mode if logger else False
    as_text = isinstance(data, str)
    threshold_mb = file_ops.large_file_threshold_mb

    def report_size(written_bytes: int) -> None:
        """Log the flushed file size without serializing the payload again."""
        data_size_mb = written_bytes / BYTES_PER_MIB
        if data_size_mb > threshold_mb and logger:
            logger.warning(
                f"Large file warning: {description} is {data_size_mb:.2f} MB. "
                f"This may cause performance issues.")

        if is_debug and logger:
            logger.debug(f"Wrote {description}: {data_size_mb:.2f} MB")

    try:
        success = (
            file_ops.write_text_with_retry(
                file_path, data, max_retries, retry_delay, sync=sync, on_written=report_size)
            if isinstance(data, str)
            else file_ops.write_json_with_retry(
                file_path, data, max_retries, retry_delay, sync=sync, on_written=report_size)
        )

        if success:
            if is_debug and logger:
                logger.debug(f"Successfully wrote {description}")
            return True
        else:
            if logger:
                logger.error(f"Failed to write {description} after {max_retries} retries. Attempting fallback save...")

            # Emergency copy next to the target; the raise that reports
            # success must stay outside this try so it is not treated as
            # a failed fallback write.
            suffix = 'txt' if as_text else 'json'
            fallback_path = f"{file_path}.fallback_{int(time.time())}.{suffix}"
            try:
                if isinstance(data, str):
                    file_ops.write_text_atomic(fallback_path, data, sync=sync)
                else:
                    file_ops.write_json_atomic(fallback_path, data, sync=sync)
            except FileDurabilityError:
                raise
            except Exception as fallback_error:
                if logger:
                    logger.error(f"Fallback save failed: {fallback_error}. DATA MAY BE LOST!")
                raise FileOperationError(f"Complete failure writing {description}") from fallback_error

            if logger:
                logger.warning(f"Data saved to fallback file: {fallback_path}")
            raise FileOperationError(f"Failed to write {description}, saved to {fallback_path}")

    except FileDurabilityError as exc:
        if logger:
            logger.error(str(exc))
        raise
    except FileOperationError:
        raise  # Re-raise FileOperationError unchanged.
    except PermissionError:
        if logger:
            logger.warning(f"Warning: No permission to write to {description}. Skipping.")
        return False
    except OSError as e:
        if logger:
            logger.warning(f"Warning: OS error writing to {description}. Error: {e}. Skipping.")
        return False
    except Exception as e:  # noqa: BLE001 - unexpected write errors are logged and skipped
        if is_debug and logger:
            logger.error(f"Exception in safe_write_file: {e}", include_traceback=True)
        if logger:
            logger.warning(f"Warning: Unexpected error writing to {description}. Skipping.")
        return False
