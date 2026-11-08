# -*- coding: utf-8 -*-
"""
Integration tests of OS-level file locks (llm_conductor/core/locking.py) on real files.

Two FileLock objects conflict even inside one process, so contention is
tested in-process; the cross-process case is in tests/system/test_cli_subprocess.py.
"""
import json
import threading

import pytest

from llm_conductor.core.locking import FileLock
from llm_conductor.exceptions import FileOperationError, LockUnavailableError


@pytest.fixture
def target(tmp_path):
    path = tmp_path / "strings.json"
    path.write_text(json.dumps({"a": "Hello"}), encoding="utf-8")
    return str(path)


def test_second_lock_on_the_same_file_is_refused(target):
    with FileLock(target), pytest.raises(LockUnavailableError) as exc:
        FileLock(target).acquire()

    assert exc.value.path == target
    assert "locked by another run" in str(exc.value)


def test_released_lock_can_be_taken_again(target):
    first = FileLock(target)
    first.acquire()
    first.release()

    with FileLock(target) as second:
        assert second.locked
    assert not second.locked


def test_locked_file_stays_readable_and_unchanged(target):
    """On Windows byte-range locks are mandatory - the locked region must not cover the content."""
    with FileLock(target), open(target, encoding="utf-8") as f:
        assert json.load(f) == {"a": "Hello"}

    with open(target, encoding="utf-8") as f:
        assert json.load(f) == {"a": "Hello"}


def test_dedicated_lock_file_and_directory_are_created(tmp_path):
    path = tmp_path / "cache" / ".llmc.lock"

    with FileLock(str(path), create=True):
        assert path.exists()


def test_missing_file_without_create_is_a_file_error(tmp_path):
    with pytest.raises(FileOperationError, match="Cannot open"):
        FileLock(str(tmp_path / "absent.json")).acquire()


def test_lock_is_released_when_the_block_raises(target):
    with pytest.raises(RuntimeError, match="boom"), FileLock(target):
        raise RuntimeError("boom")

    with FileLock(target):
        pass


def test_acquiring_twice_through_one_object_is_a_programming_error(target):
    lock = FileLock(target)
    lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="already locked"):
            lock.acquire()
    finally:
        lock.release()


def test_release_without_acquire_is_a_no_op(target):
    FileLock(target).release()


def test_repr_does_not_take_the_state_lock(target):
    lock = FileLock(target)
    with lock._state_lock:
        assert "locked=False" in repr(lock)


def test_only_one_of_many_threads_gets_the_lock(target):
    taken = []
    start = threading.Barrier(8, timeout=5)
    attempted = threading.Barrier(8, timeout=5)

    def contend():
        lock = FileLock(target)
        start.wait()
        try:
            lock.acquire()
            taken.append(True)
        except LockUnavailableError:
            taken.append(False)
        attempted.wait()          # the winner holds the lock until everyone has tried
        lock.release()

    threads = [threading.Thread(target=contend) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert taken.count(True) == 1 and len(taken) == 8
