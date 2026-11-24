"""Resume persistence, invalid indexes, and recoverable flush failures."""
import json

import pytest

from llm_conductor.constants import RESUME_SCHEMA_VERSION
from llm_conductor.core import resume as resume_module
from llm_conductor.core.io import FileOperations
from llm_conductor.core.resume import (
    ResumeIndex,
    ResumeLeafEntry,
    StageResumeSession,
    load_resume_index,
    resume_index_to_dict,
)
from llm_conductor.exceptions import FileOperationError
from tests.support import RecordingLogger


@pytest.fixture
def session(tmp_path):
    return StageResumeSession(
        source={'a': 'one', 'b': 'two'}, stage='translated', operation='translation',
        relative_path='sample.json', run_fingerprint='fp',
        stage_path=str(tmp_path / 'stage.json'), index_path=str(tmp_path / 'index.json'),
        file_ops=FileOperations(), logger=RecordingLogger(), flush_interval=2,
        continue_mode=True,
        prior=ResumeIndex(RESUME_SCHEMA_VERSION, 'translated', 'translation', 'fp', 'sample.json', {
            '["a"]': ResumeLeafEntry('key-a', 'first'),
            '["b"]': ResumeLeafEntry('key-b', 'second'),
        }),
    )


def test_missing_index_is_optional_but_invalid_json_is_not(tmp_path):
    path = tmp_path / 'index.json'
    logger = RecordingLogger()
    assert load_resume_index(str(path), FileOperations(), logger) is None
    assert not logger.records
    path.write_text('{', encoding='utf-8')
    with pytest.raises(FileOperationError, match='invalid JSON'):
        load_resume_index(str(path), FileOperations())


@pytest.mark.parametrize('name', ['index.json', 'index', 'index.data'])
def test_resume_index_is_json_regardless_of_target_extension(session, tmp_path, name):
    path = str(tmp_path / name)
    expected = session.snapshot_index()
    session.file_ops.write_json_atomic(path, resume_index_to_dict(expected))
    assert load_resume_index(path, session.file_ops) == expected


@pytest.mark.parametrize(('field', 'value'), [
    ('schema_version', 999), ('stage', ''), ('operation', None),
    ('run_fingerprint', ''), ('relative_path', None), ('leaves', []),
])
def test_invalid_index_header_is_refused(session, field, value):
    document = resume_index_to_dict(session.snapshot_index())
    document[field] = value
    session.file_ops.write_json_atomic(session.index_path, document)
    with pytest.raises(FileOperationError):
        load_resume_index(session.index_path, session.file_ops)


@pytest.mark.parametrize('document', [None, [], 'not an object'])
def test_non_object_index_is_refused(tmp_path, document):
    path = tmp_path / 'index.json'
    path.write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(FileOperationError, match='not a JSON object'):
        load_resume_index(str(path), FileOperations())


def test_corrupt_leaf_entries_are_never_reused(session):
    document = resume_index_to_dict(session.snapshot_index())
    document['leaves'] = {'["a"]': None, '["b"]': {'cache_key': 'key-b', 'value': 42}}
    session.file_ops.write_json_atomic(session.index_path, document)
    assert load_resume_index(session.index_path, session.file_ops).leaves == {}


def test_flush_retains_unvisited_prior_and_drops_mismatched_or_skipped_leaves(session):
    session.note_accepted('["a"]', 'key-a', 'first')
    session.flush_for_shutdown()
    restored = load_resume_index(session.index_path, session.file_ops)
    assert len(restored.leaves) == 2
    assert session.file_ops.read_json(session.stage_path) == {'a': 'first', 'b': 'second'}
    assert session.lookup('["b"]', 'changed-key') is None
    session.discard_prior('["a"]')
    session.flush()
    assert set(load_resume_index(session.index_path, session.file_ops).leaves) == {'["a"]'}


def test_discard_skipped_prior_candidate(session):
    session.discard_prior('["b"]')
    session.flush()
    assert set(load_resume_index(session.index_path, session.file_ops).leaves) == {'["a"]'}


@pytest.mark.parametrize('error', [FileOperationError('full'), OSError('unavailable'), None])
def test_periodic_flush_failure_keeps_accepted_values_for_later_retry(session, monkeypatch, error):
    original = session.flush

    def fail():
        if error is not None:
            raise error
        return False

    monkeypatch.setattr(session, 'flush', fail)
    session.note_accepted('["a"]', 'key-a', 'first')
    session.note_accepted('["b"]', 'key-b', 'second')
    assert session.logger.messages('WARNING')
    monkeypatch.setattr(session, 'flush', original)
    assert session.flush()
    assert len(load_resume_index(session.index_path, session.file_ops).leaves) == 2


def test_shutdown_flush_error_is_logged(session, monkeypatch):
    def fail(*args, **kwargs):
        raise FileOperationError('full')

    monkeypatch.setattr(resume_module, 'safe_write_file', fail)
    session.flush_for_shutdown()
    assert session.logger.contains('Failed to flush stage resume during shutdown', 'WARNING')
