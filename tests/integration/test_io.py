# -*- coding: utf-8 -*-
"""
Integration tests for the file layer (llm_conductor/core/io.py) on tmp_path:
atomic write, retry, fallback save, safe_read_file/safe_write_file.
"""
import json

import pytest

from llm_conductor.core import io as io_module
from llm_conductor.core.io import FileOperations, safe_read_file, safe_write_file
from llm_conductor.exceptions import FileOperationError
from tests.support import RecordingLogger


@pytest.fixture
def file_ops():
    return FileOperations(logger=RecordingLogger())


def _temp_leftovers(directory):
    return [p.name for p in directory.iterdir() if p.name.endswith('.tmp')]


# ============================================================================
# FileOperations — atomic write
# ============================================================================

def test_json_roundtrip_keeps_cyrillic_unescaped(tmp_path, file_ops):
    target = tmp_path / "out.json"

    file_ops.write_json_atomic(str(target), {"greeting": "Привет", "n": [1, None]})

    assert file_ops.read_json(str(target)) == {"greeting": "Привет", "n": [1, None]}
    assert "Привет" in target.read_text(encoding="utf-8")     # ensure_ascii=False
    assert _temp_leftovers(tmp_path) == []


def test_atomic_write_replaces_existing_file(tmp_path, file_ops):
    target = tmp_path / "out.json"
    target.write_text('{"old": true}', encoding="utf-8")

    file_ops.write_json_atomic(str(target), {"new": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"new": True}


def test_failed_serialization_keeps_original_and_removes_temp_file(tmp_path, file_ops):
    target = tmp_path / "out.json"
    target.write_text('{"old": true}', encoding="utf-8")

    with pytest.raises(TypeError):
        file_ops.write_json_atomic(str(target), {"bad": object()})

    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    assert _temp_leftovers(tmp_path) == []


def test_text_roundtrip_is_byte_exact(tmp_path, file_ops):
    target = tmp_path / "llm.toml"
    content = "system = '''\nYou are a translator {0} <b>\n'''\n"

    file_ops.write_text_atomic(str(target), content)

    assert file_ops.read_text_with_retry(str(target)) == content


# ============================================================================
# FileOperations — retry
# ============================================================================

def _flaky(fail_times, error):
    """Stub for run_with_timeout: the first fail_times calls raise error."""
    state = {'calls': 0}

    def fake_run_with_timeout(func, timeout_seconds, operation_name="operation"):
        state['calls'] += 1
        if state['calls'] <= fail_times:
            raise error
        return func()

    return fake_run_with_timeout, state


def test_read_text_retries_transient_error_with_backoff(tmp_path, file_ops, monkeypatch, no_sleep):
    target = tmp_path / "llm.toml"
    target.write_text("ok", encoding="utf-8")
    fake, state = _flaky(2, PermissionError("locked by another process"))
    monkeypatch.setattr(io_module, 'run_with_timeout', fake)

    assert file_ops.read_text_with_retry(str(target)) == "ok"
    assert state['calls'] == 3
    assert no_sleep == [1.0, 2.0]


def test_write_json_with_retry_recovers_after_transient_error(tmp_path, file_ops, monkeypatch, no_sleep):
    target = tmp_path / "out.json"
    fake, _ = _flaky(1, PermissionError("locked"))
    monkeypatch.setattr(io_module, 'run_with_timeout', fake)

    assert file_ops.write_json_with_retry(str(target), {"a": 1}) is True
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert no_sleep == [1.0]


def test_write_json_with_retry_falls_back_to_next_encoding(tmp_path, file_ops, monkeypatch, no_sleep):
    used = []
    original = file_ops.write_json_atomic

    def picky_write(path, data, encoding='utf-8'):
        used.append(encoding)
        if encoding == 'utf-8':
            raise UnicodeEncodeError('utf-8', 'x', 0, 1, 'test')
        original(path, data, encoding)

    monkeypatch.setattr(file_ops, 'write_json_atomic', picky_write)

    assert file_ops.write_json_with_retry(str(tmp_path / "out.json"), {"a": 1}) is True
    assert used == ['utf-8', 'utf-8-sig']
    assert no_sleep == []                      # encoding switch — no wait


def test_task51_missing_text_file_is_not_retried(tmp_path, file_ops, no_sleep):
    with pytest.raises(FileNotFoundError):
        file_ops.read_text_with_retry(str(tmp_path / "missing.toml"))

    assert no_sleep == []


def test_task51_missing_directory_fails_without_retries_or_encoding_loop(tmp_path, file_ops, no_sleep):
    """Permanent path errors must not burn retries or walk the encoding list."""
    assert file_ops.write_json_with_retry(str(tmp_path / "no_dir" / "out.json"), {"a": 1}) is False

    assert no_sleep == []


# ============================================================================
# safe_read_file
# ============================================================================

def test_safe_read_detects_mode_by_extension(tmp_path, file_ops):
    (tmp_path / "data.json").write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "llm.toml").write_text('a = 1', encoding="utf-8")

    assert safe_read_file(str(tmp_path / "data.json"), "data", file_ops) == {"a": 1}
    assert safe_read_file(str(tmp_path / "llm.toml"), "profile", file_ops) == "a = 1"


def test_safe_read_as_text_overrides_extension(tmp_path, file_ops):
    (tmp_path / "data.json").write_text('{"a": 1}', encoding="utf-8")

    assert safe_read_file(str(tmp_path / "data.json"), "data", file_ops, as_text=True) == '{"a": 1}'


@pytest.mark.parametrize("content, reason", [
    (None, "not found"),
    ("{broken", "invalid JSON"),
    (b"\xcf\xf0\xe8\xe2\xe5\xf2", "decoding error"),      # Cyrillic greeting bytes in cp1251
])
def test_safe_read_critical_failure_raises(tmp_path, file_ops, content, reason):
    path = tmp_path / "input.json"
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif content is not None:
        path.write_text(content, encoding="utf-8")
    logger = RecordingLogger()

    with pytest.raises(FileOperationError, match=reason):
        safe_read_file(str(path), "input data", file_ops, logger, critical=True)

    assert logger.contains(reason, level='ERROR')


def test_safe_read_non_critical_failure_returns_none_with_warning(tmp_path, file_ops):
    logger = RecordingLogger()

    result = safe_read_file(str(tmp_path / "glossary.json"), "glossary", file_ops, logger, critical=False)

    assert result is None
    assert logger.contains("Proceeding without glossary", level='WARNING')


# ============================================================================
# safe_write_file
# ============================================================================

def test_safe_write_dict_as_json_and_str_as_text(tmp_path, file_ops):
    assert safe_write_file(str(tmp_path / "a.json"), {"k": "в"}, "a", file_ops) is True
    assert safe_write_file(str(tmp_path / "b.toml"), "k = 'в'\n", "b", file_ops) is True

    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8")) == {"k": "в"}
    assert (tmp_path / "b.toml").read_text(encoding="utf-8") == "k = 'в'\n"


def _break_primary_write(monkeypatch, file_ops):
    monkeypatch.setattr(file_ops, 'write_json_with_retry', lambda *a, **kw: False)
    monkeypatch.setattr(file_ops, 'write_text_with_retry', lambda *a, **kw: False)


@pytest.mark.parametrize("data, suffix", [({"k": "v"}, "json"), ("raw text", "txt")])
def test_safe_write_failure_saves_fallback_and_raises(tmp_path, file_ops, monkeypatch, data, suffix):
    _break_primary_write(monkeypatch, file_ops)
    logger = RecordingLogger()

    with pytest.raises(FileOperationError):
        safe_write_file(str(tmp_path / "out"), data, "final output", file_ops, logger)

    fallbacks = list(tmp_path.glob(f"out.fallback_*.{suffix}"))
    assert len(fallbacks) == 1
    restored = fallbacks[0].read_text(encoding="utf-8")
    assert (json.loads(restored) if suffix == "json" else restored) == data
    assert logger.contains("Data saved to fallback file", level='WARNING')


def test_task52_successful_fallback_is_not_reported_as_data_loss(tmp_path, file_ops, monkeypatch):
    _break_primary_write(monkeypatch, file_ops)
    logger = RecordingLogger()

    with pytest.raises(FileOperationError, match="saved to"):
        safe_write_file(str(tmp_path / "out"), {"k": "v"}, "final output", file_ops, logger)

    assert not logger.contains("DATA MAY BE LOST")
