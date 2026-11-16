# -*- coding: utf-8 -*-
"""Unit tests for HeartbeatWriter and count_string_leaves (Task #16)."""
import builtins
import io
import json
import threading
import time
from pathlib import Path

import pytest

from llm_conductor.cli import heartbeat as heartbeat_module
from llm_conductor.cli.exit_status import ExitRecord
from llm_conductor.cli.glossary import GlossarySource
from llm_conductor.cli.heartbeat import HeartbeatWriter
from llm_conductor.cli.targets import RunTarget, TargetEntry
from llm_conductor.constants import (
    EXIT_FAILURE_RATIO_EXCEEDED,
    EXIT_INTERRUPTED,
    EXIT_SUCCESS,
    HEARTBEAT_ACTIVITY_WAITING_API,
    HEARTBEAT_SCHEMA_VERSION,
    HEARTBEAT_STATUS_COMPLETED,
    HEARTBEAT_STATUS_INTERRUPTED,
    HEARTBEAT_STATUS_RUNNING,
    OPERATION_TRANSLATION,
    PROGRESS_STATE_FILENAME,
)
from llm_conductor.core import io as io_module
from llm_conductor.core.io import FileOperations
from llm_conductor.exceptions import FileOperationError
from llm_conductor.l10n.processor import count_string_leaves
from tests.support import RecordingLogger, read_heartbeat_text


def _target(tmp_path: Path, names: list[str]) -> RunTarget:
    entries = tuple(
        TargetEntry(str(tmp_path / name), name) for name in names
    )
    return RunTarget(str(tmp_path / "ui.toml"), entries)


def _writer(tmp_path: Path, *, interval: int = 100, interval_sec: float = 0,
            names: list[str] | None = None,
            glossary: GlossarySource | None = None) -> HeartbeatWriter:
    cache_dir = str(tmp_path / "cache")
    return HeartbeatWriter(
        cache_dir=cache_dir,
        interval=interval,
        interval_sec=interval_sec,
        file_ops=FileOperations(timeout=5, large_file_threshold_mb=100),
        logger=RecordingLogger(),
        target=_target(tmp_path, names or ["a.json"]),
        profile_name="test-profile",
        stages=(OPERATION_TRANSLATION,),
        glossary=glossary or GlossarySource(terms={}, path=None, origin=None),
    )


def _read(writer: HeartbeatWriter) -> dict:
    text = read_heartbeat_text(Path(writer.path))
    assert "\n" not in text  # compact JSON
    return json.loads(text)


@pytest.mark.parametrize('permanent', [False, True])
def test_heartbeat_reader_handles_sharing_violations(tmp_path, monkeypatch, no_sleep, permanent):
    writer = _writer(tmp_path)
    writer.start()
    real_open = builtins.open
    attempts = []
    lock = threading.Lock()

    def locked_open(file, *args, **kwargs):
        if file == writer.path or file == Path(writer.path):
            with lock:
                attempts.append(file)
                if permanent or len(attempts) == 1:
                    raise PermissionError('simulated heartbeat sharing violation')
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, 'open', locked_open)
    monkeypatch.setattr(io, 'open', locked_open)
    if permanent:
        with pytest.raises(FileOperationError, match='permission'):
            _read(writer)
        assert len(attempts) == 3
    else:
        assert _read(writer)['status'] == HEARTBEAT_STATUS_RUNNING
        assert len(attempts) == 2


def test_start_writes_running_snapshot_with_null_exit_code(tmp_path):
    writer = _writer(tmp_path)
    writer.start()

    snap = _read(writer)
    assert snap["schema_version"] == HEARTBEAT_SCHEMA_VERSION
    assert snap["status"] == HEARTBEAT_STATUS_RUNNING
    assert snap["exit_code"] is None
    assert snap["profile"] == {"name": "test-profile"}
    assert snap["stages"] == [OPERATION_TRANSLATION]
    assert snap["progress"]["files_total"] == 1
    assert snap["progress"]["file_index"] == 0
    assert snap["progress"]["percent"] == 0.0
    assert Path(writer.path).name == PROGRESS_STATE_FILENAME


def test_heartbeat_uses_atomic_write_without_file_or_directory_sync(tmp_path, monkeypatch):
    writer = _writer(tmp_path)

    def unexpected_sync(*args):
        pytest.fail("Heartbeat must not request durable storage")

    monkeypatch.setattr(io_module.os, "fsync", unexpected_sync)
    monkeypatch.setattr(writer._file_ops, "_sync_parent_directory", unexpected_sync)
    writer.start()
    writer.finalize(HEARTBEAT_STATUS_COMPLETED, EXIT_SUCCESS)
    assert _read(writer)["status"] == HEARTBEAT_STATUS_COMPLETED


def test_hybrid_percent_across_two_files(tmp_path):
    writer = _writer(tmp_path, names=["a.json", "b.json"], interval=1)
    writer.start()
    a = TargetEntry(str(tmp_path / "a.json"), "a.json")
    writer.on_file_start(a, 1)
    writer.begin_stage(OPERATION_TRANSLATION, leaves_total=2)
    writer.note_leaf()

    snap = _read(writer)
    # files_done=0, leaf_frac=0.5 → 100 * 0.5 / 2 = 25
    assert snap["progress"]["percent"] == 25.0
    assert snap["progress"]["leaves_done"] == 1
    assert snap["progress"]["leaves_total"] == 2

    writer.note_leaf()
    writer.on_file_done("a.json", "done")
    snap = _read(writer)
    assert snap["progress"]["files_done"] == 1
    assert snap["progress"]["percent"] == 50.0
    assert snap["files"]["done"] == ["a.json"]
    assert snap["files"]["pending"] == ["b.json"]


def test_interval_zero_skips_leaf_only_flushes(tmp_path):
    writer = _writer(tmp_path, interval=0)
    writer.start()
    writer.begin_stage(OPERATION_TRANSLATION, leaves_total=3)
    before = Path(writer.path).read_text(encoding="utf-8")
    writer.note_leaf()
    writer.note_leaf()
    after = Path(writer.path).read_text(encoding="utf-8")
    assert before == after
    assert json.loads(after)["progress"]["leaves_done"] == 0  # not flushed


def test_interval_flushes_every_n_leaves(tmp_path):
    writer = _writer(tmp_path, interval=2)
    writer.start()
    writer.begin_stage(OPERATION_TRANSLATION, leaves_total=4)
    writer.note_leaf()
    assert _read(writer)["progress"]["leaves_done"] == 0
    writer.note_leaf()
    assert _read(writer)["progress"]["leaves_done"] == 2


def test_finalize_completed_sets_exit_code_and_problems(tmp_path):
    writer = _writer(tmp_path)
    writer.start()
    writer.set_problems([ExitRecord(EXIT_FAILURE_RATIO_EXCEEDED, "too many failures")])
    writer.finalize(HEARTBEAT_STATUS_COMPLETED, EXIT_FAILURE_RATIO_EXCEEDED)

    snap = _read(writer)
    assert snap["status"] == HEARTBEAT_STATUS_COMPLETED
    assert snap["exit_code"] == EXIT_FAILURE_RATIO_EXCEEDED
    assert snap["problems"] == [
        {"code": EXIT_FAILURE_RATIO_EXCEEDED, "reason": "too many failures"}
    ]
    # Second finalize is a no-op
    writer.finalize(HEARTBEAT_STATUS_INTERRUPTED, EXIT_INTERRUPTED)
    assert _read(writer)["status"] == HEARTBEAT_STATUS_COMPLETED


def test_finalize_interrupted(tmp_path):
    writer = _writer(tmp_path)
    writer.start()
    writer.finalize(HEARTBEAT_STATUS_INTERRUPTED, EXIT_INTERRUPTED)
    snap = _read(writer)
    assert snap["status"] == HEARTBEAT_STATUS_INTERRUPTED
    assert snap["exit_code"] == EXIT_INTERRUPTED


@pytest.mark.parametrize("status", ["failed", "crashed", "aborted"])
def test_non_done_file_appears_in_files_failed(tmp_path, status):
    writer = _writer(tmp_path, names=["a.json", "b.json"], interval=0)
    writer.start()
    writer.on_file_start(TargetEntry(str(tmp_path / "a.json"), "a.json"), 1)
    writer.on_file_done("a.json", status)

    snap = _read(writer)
    assert snap["files"]["failed"] == [{"path": "a.json", "status": status}]
    assert snap["files"]["done"] == []
    assert snap["files"]["pending"] == ["b.json"]
    assert snap["progress"]["files_done"] == 1


def test_glossary_fields_mirror_source(tmp_path):
    glossary = GlossarySource(
        terms={"Guild": "Гильдия"},
        path=str(tmp_path / "glossary.json"),
        origin="next to the target",
    )
    writer = _writer(tmp_path, glossary=glossary)
    writer.start()
    snap = _read(writer)
    assert snap["glossary"] == {
        "origin": "next to the target",
        "path": str(tmp_path / "glossary.json"),
        "term_count": 1,
    }


def test_write_failure_does_not_raise(tmp_path, monkeypatch):
    writer = _writer(tmp_path)
    logger = writer._logger

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("llm_conductor.cli.heartbeat.safe_write_file", boom)
    writer.start()  # must not raise
    assert logger.contains("Heartbeat write", level="WARNING")


@pytest.mark.parametrize("tree, max_depth, expected", [
    ("hi", 30, 1),
    ({"a": "x", "b": "  ", "c": 1}, 30, 2),
    (["a", ["b", "c"]], 30, 3),
    ({"deep": {"x": "y"}}, 0, 0),  # child dict at depth 1 > 0
    ({"deep": {"x": "y"}}, 1, 0),  # string at depth 2 > 1
    ({"deep": {"x": "y"}}, 2, 1),
])
def test_count_string_leaves(tree, max_depth, expected):
    assert count_string_leaves(tree, max_depth) == expected


def test_time_pulse_refreshes_the_clock_without_moving_progress(tmp_path):
    writer = _writer(tmp_path, interval_sec=0.2)
    try:
        writer.start()
        writer.begin_stage(OPERATION_TRANSLATION, leaves_total=2)
        writer.set_activity(HEARTBEAT_ACTIVITY_WAITING_API)
        first = _read(writer)
        later = first
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            time.sleep(0.05)
            later = _read(writer)
            if (later["updated_at"] != first["updated_at"]
                    and later["activity"] == HEARTBEAT_ACTIVITY_WAITING_API):
                break
        assert later["activity"] == HEARTBEAT_ACTIVITY_WAITING_API
        assert later["updated_at"] != first["updated_at"]
        assert later["elapsed_sec"] >= first["elapsed_sec"]
        assert later["progress"]["leaves_done"] == first["progress"]["leaves_done"] == 0
        assert later["last_progress_at"] == first["last_progress_at"] is None
    finally:
        writer.finalize(HEARTBEAT_STATUS_COMPLETED, EXIT_SUCCESS)


@pytest.mark.parametrize("status, code", [
    (HEARTBEAT_STATUS_COMPLETED, EXIT_SUCCESS),
    (HEARTBEAT_STATUS_COMPLETED, EXIT_FAILURE_RATIO_EXCEEDED),
    (HEARTBEAT_STATUS_INTERRUPTED, EXIT_INTERRUPTED),
])
def test_pulse_does_not_rewrite_a_terminal_snapshot(tmp_path, status, code):
    writer = _writer(tmp_path, interval_sec=0.1)
    writer.start()
    writer.finalize(status, code)

    text = Path(writer.path).read_text(encoding="utf-8")
    time.sleep(0.45)

    assert Path(writer.path).read_text(encoding="utf-8") == text
    snap = json.loads(text)
    assert snap["status"] == status
    assert snap["exit_code"] == code


def test_pulse_cannot_replace_a_newer_snapshot(tmp_path, monkeypatch):
    writer = _writer(tmp_path, interval=1, interval_sec=0.02)
    real = heartbeat_module.safe_write_file
    counts: list[int] = []

    def recording(path, text, *args, **kwargs):
        counts.append(json.loads(text)["progress"]["leaves_done"])
        time.sleep(0.01)
        return real(path, text, *args, **kwargs)

    monkeypatch.setattr(heartbeat_module, "safe_write_file", recording)
    try:
        writer.start()
        writer.begin_stage(OPERATION_TRANSLATION, leaves_total=8)

        def hammer():
            for _ in range(8):
                writer.note_leaf()

        thread = threading.Thread(target=hammer)
        thread.start()
        thread.join()
        writer.finalize(HEARTBEAT_STATUS_COMPLETED, EXIT_SUCCESS)
    finally:
        if not writer.finalized:
            writer.finalize(HEARTBEAT_STATUS_COMPLETED, EXIT_SUCCESS)

    assert counts == sorted(counts)
    snap = _read(writer)
    assert snap["status"] == HEARTBEAT_STATUS_COMPLETED
    assert snap["progress"]["leaves_done"] == 8
    assert snap["last_progress_at"] is not None


def test_successful_finalize_exit_zero(tmp_path):
    writer = _writer(tmp_path)
    writer.start()
    writer.on_file_start(TargetEntry(str(tmp_path / "a.json"), "a.json"), 1)
    writer.on_file_done("a.json", "done")
    writer.finalize(HEARTBEAT_STATUS_COMPLETED, EXIT_SUCCESS)
    snap = _read(writer)
    assert snap["exit_code"] == EXIT_SUCCESS
    assert snap["progress"]["percent"] == 100.0
