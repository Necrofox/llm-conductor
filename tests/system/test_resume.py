# -*- coding: utf-8 -*-
"""
System tests for stage-local resume (Issue #4 MVP): SIGINT flush, --continue,
fingerprint hard-fail, and degraded leaves staying out of the index.
"""
import json
import os
import signal
import subprocess
import sys
from textwrap import dedent

import pytest

from llm_conductor.constants import (
    EXIT_FAILURE_RATIO_EXCEEDED,
    EXIT_INTERRUPTED,
    EXIT_SUCCESS,
    OPERATION_TRANSLATION,
    RESUME_SCHEMA_VERSION,
)
from llm_conductor.core import shutdown
from llm_conductor.core.cache import CacheManager
from llm_conductor.core.resume import StageResumeSession
from llm_conductor.exceptions import FileOperationError, RunInterrupted
from tests.system.harness import REPO_ROOT, TRANSLATE_PROMPT

RUN = ("run", "--target", "sample.json")
TRANSLATE_NO_CACHE = (*RUN, "--stage", "translate", "--no-cache")


def _interrupt_on(model, call_number: int):
    def interrupt(n):
        if n == call_number:
            shutdown.signal_handler(signal.SIGINT, None)

    model.on_model_call = interrupt


def _reset_shutdown():
    """Allow a second run_main in the same test after an interrupt."""
    manager = shutdown._shutdown_manager
    with manager._lock:
        manager._shutdown_requested = False
        manager._shutdown_in_progress = False
        manager._config_instance = None


def test_sigint_with_no_cache_writes_partial_stage_and_resume_index(
        workspace, model, run_main):
    _interrupt_on(model, 3)

    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED

    assert len(model.model_calls) == 3
    assert workspace.exists(workspace.result("translated"))
    assert not workspace.exists(workspace.result("final"))
    resume_path = workspace.result("resume/translated")
    assert workspace.exists(resume_path)
    index = workspace.read_json(resume_path)
    assert index["schema_version"] == RESUME_SCHEMA_VERSION
    assert index["stage"] == "translated"
    assert len(index["leaves"]) == 2
    translated = workspace.read_json(workspace.result("translated"))
    # Accepted leaves are overlaid; the interrupted leaf stays English.
    assert translated["ui"]["title"].startswith("RU(")
    assert translated["ui"]["gold_counter"].startswith("RU(")


def test_continue_no_cache_skips_accepted_leaves(workspace, model, run_main):
    _interrupt_on(model, 3)
    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED
    first_calls = len(model.model_calls)
    assert first_calls == 3

    model.on_model_call = None
    model.requests.clear()
    _reset_shutdown()

    assert run_main(*TRANSLATE_NO_CACHE, "--continue") == EXIT_SUCCESS

    # SAMPLE has 7 eligible string leaves; two reused from resume → 5 API calls.
    assert first_calls == 3
    assert len(model.model_calls) == 5
    final = workspace.read_json(workspace.result("translated"))
    assert final["ui"]["title"] == "RU(Welcome to the Guild)"
    assert final["ui"]["gold_counter"] == "RU(You have {0} gold coins.)"
    assert final["options"][0] == "RU(Easy)"
    usage = workspace.read_json('result/sample.json/usage.json')
    assert usage['non_api'][OPERATION_TRANSLATION]['resume'] == 2


@pytest.mark.parametrize('stage', ['translate', 'verify'])
def test_no_cache_interrupt_then_continue_populates_cache_for_normal_runs(workspace, model, run_main, stage):
    if stage == 'verify':
        assert run_main(*TRANSLATE_NO_CACHE) == EXIT_SUCCESS
        model.requests.clear()
    args = (*RUN, '--stage', stage)
    operation = 'translation' if stage == 'translate' else 'verification'
    directory = 'translated' if stage == 'translate' else 'redacted'
    _interrupt_on(model, 3)
    assert run_main(*args, '--no-cache') == EXIT_INTERRUPTED
    prior = workspace.read_json(workspace.result(f'resume/{directory}'))['leaves']
    assert len(prior) == 2
    assert not workspace.exists(f'cache/{operation}_cache.json')

    _reset_shutdown()
    model.on_model_call = None
    model.requests.clear()
    assert run_main(*args, '--continue') == EXIT_SUCCESS
    assert len(model.model_calls) == 5
    usage = workspace.read_json('result/sample.json/usage.json')
    assert usage['non_api'][operation]['resume'] == 2
    assert usage['non_api'][operation]['cache_hit'] == 0
    assert len([attempt for attempt in usage['attempts'] if attempt['stage'] == operation]) == 5
    cache = workspace.read_json(f'cache/{operation}_cache.json')
    for entry in prior.values():
        stored = cache[entry['cache_key']]
        assert stored['value'] == entry['value']
        assert stored['fingerprints']
        assert stored['value_hash']
        assert stored['original_key'][0]

    final = workspace.read_json(workspace.result('final'))
    model.requests.clear()
    assert run_main(*args) == EXIT_SUCCESS
    assert not model.model_calls
    assert workspace.read_json(workspace.result('final')) == final
    usage = workspace.read_json('result/sample.json/usage.json')
    assert usage['non_api'][operation]['resume'] == 0
    assert usage['non_api'][operation]['cache_hit'] == 7


@pytest.mark.parametrize('disabled_by', ['cli', 'config'])
def test_disabled_cache_is_not_populated_by_resume(workspace, model, run_main, monkeypatch, disabled_by):
    _interrupt_on(model, 3)
    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED
    _reset_shutdown()
    model.on_model_call = None
    model.requests.clear()
    if disabled_by == 'config':
        workspace.write_configs(cache__enabled=False)
    flags = ('--no-cache',) if disabled_by == 'cli' else ()

    def forbidden(*args, **kwargs):
        pytest.fail('Disabled cache must not be populated by resumed or new answers')

    monkeypatch.setattr(CacheManager, 'set', forbidden)
    assert run_main(*RUN, '--stage', 'translate', '--continue', *flags) == EXIT_SUCCESS
    assert len(model.model_calls) == 5
    assert not workspace.exists('cache/translation_cache.json')


def test_resumed_answers_participate_in_cache_autosave_before_another_interrupt(workspace, model, run_main):
    workspace.write_configs(cache__auto_save_interval=1)
    _interrupt_on(model, 3)
    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED
    prior = workspace.read_json(workspace.result('resume/translated'))['leaves']
    _reset_shutdown()

    def interrupt(_):
        # This is the first non-resumed leaf, before its reply is accepted.
        cache = workspace.read_json('cache/translation_cache.json')
        assert {key: cache[key]['value'] for key in cache} == {
            entry['cache_key']: entry['value'] for entry in prior.values()
        }
        shutdown.signal_handler(signal.SIGINT, None)

    model.requests.clear()
    model.on_model_call = interrupt
    assert run_main(*RUN, '--stage', 'translate', '--continue') == EXIT_INTERRUPTED


def test_without_continue_does_not_reuse_resume_index(workspace, model, run_main):
    _interrupt_on(model, 3)
    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED

    model.on_model_call = None
    model.requests.clear()
    _reset_shutdown()

    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_SUCCESS
    # Full rewalk — no silent auto-resume.
    assert len(model.model_calls) == 7


def test_continue_fingerprint_mismatch_refuses_stale_index(workspace, model, run_main, caplog):
    _interrupt_on(model, 3)
    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED

    model.on_model_call = None
    model.requests.clear()
    _reset_shutdown()
    # Prompt change → run_fingerprint mismatch.
    workspace.write_configs(
        translation__prompts__style=TRANSLATE_PROMPT + " (revised)",
    )

    code = run_main(*TRANSLATE_NO_CACHE, "--continue")
    assert code == EXIT_FAILURE_RATIO_EXCEEDED
    assert "fingerprint does not match" in caplog.text
    assert len(model.model_calls) == 0


def test_continue_retries_degraded_leaf_not_in_index(workspace, model, run_main):
    model.reject_leaf_substrings = ("Welcome to the Guild",)
    _interrupt_on(model, 4)
    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED

    resume = workspace.read_json(workspace.result("resume/translated"))
    title_key = json.dumps(["ui", "title"], separators=(",", ":"))
    assert title_key not in resume["leaves"]
    # Call 1 rejected title; calls 2–3 accepted; call 4 interrupted.
    assert len(resume["leaves"]) == 2

    model.on_model_call = None
    model.reject_leaf_substrings = ()
    model.requests.clear()
    _reset_shutdown()

    assert run_main(*TRANSLATE_NO_CACHE, "--continue") == EXIT_SUCCESS
    title_calls = [
        body for body in model.model_calls
        if "Welcome to the Guild" in body["messages"][-1]["content"]
    ]
    assert len(title_calls) == 1
    translated = workspace.read_json(workspace.result("translated"))
    assert translated["ui"]["title"] == "RU(Welcome to the Guild)"


@pytest.mark.parametrize('stage', ['translate', 'all'])
def test_continue_without_index_starts_fresh(workspace, model, run_main, stage):
    assert run_main(*RUN, '--no-cache', '--stage', stage, '--continue') == EXIT_SUCCESS
    assert len(model.model_calls) == (7 if stage == 'translate' else 14)


def test_continue_all_starts_verify_after_interrupted_translation(workspace, model, run_main):
    _interrupt_on(model, 3)
    assert run_main(*RUN, '--no-cache') == EXIT_INTERRUPTED
    _reset_shutdown()
    model.on_model_call = None
    model.requests.clear()
    assert run_main(*RUN, '--no-cache', '--continue') == EXIT_SUCCESS
    assert len(model.model_calls) == 12
    assert workspace.exists(workspace.result('redacted'))


@pytest.mark.parametrize('stage', ['all', 'verify'])
def test_continue_verification_reuses_only_verified_leaves(workspace, model, run_main, stage):
    _interrupt_on(model, 10)
    assert run_main(*RUN, '--no-cache') == EXIT_INTERRUPTED
    _reset_shutdown()
    model.on_model_call = None
    model.requests.clear()
    assert run_main(*RUN, '--no-cache', '--stage', stage, '--continue') == EXIT_SUCCESS
    assert len(model.model_calls) == 5
    assert workspace.read_json(workspace.result('final'))['ui']['title'].startswith('OK(')


def test_continue_manifest_processes_files_not_started_before_interrupt(workspace, model, run_main):
    workspace.write_json('later.json', {'text': 'Later file'})
    workspace.write_text('job.toml', '[[targets]]\noriginal="sample.json"\n'
                         '[[targets]]\noriginal="later.json"\n')
    args = ('run', '--target', 'job.toml', '--no-cache')
    _interrupt_on(model, 3)
    assert run_main(*args) == EXIT_INTERRUPTED
    _reset_shutdown()
    model.on_model_call = None
    model.requests.clear()
    assert run_main(*args, '--continue') == EXIT_SUCCESS
    assert len(model.model_calls) == 14
    assert workspace.exists(workspace.result('final', relative='later.json', target='job.toml'))


def test_repeated_interrupt_retains_unvisited_prior_progress(workspace, model, run_main, monkeypatch):
    _interrupt_on(model, 3)
    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED
    path = workspace.result('resume/translated')
    before = workspace.read_json(path)['leaves']
    _reset_shutdown()
    model.on_model_call = None
    original = StageResumeSession.lookup
    calls = 0

    def lookup(self, *args):
        nonlocal calls
        calls += 1
        if calls == 2:
            shutdown.signal_handler(signal.SIGINT, None)
        return original(self, *args)

    monkeypatch.setattr(StageResumeSession, 'lookup', lookup)
    assert run_main(*TRANSLATE_NO_CACHE, '--continue') == EXIT_INTERRUPTED
    assert workspace.read_json(path)['leaves'] == before
    assert workspace.read_json(workspace.result('translated'))['ui']['gold_counter'].startswith('RU(')


def test_periodic_write_failure_preserves_accepted_answer(workspace, model, run_main, monkeypatch, caplog):
    workspace.write_json('sample.json', {str(i): f'Unique text {i}' for i in range(12)})
    original = StageResumeSession.flush
    calls = 0

    def flush(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FileOperationError('injected disk failure')
        return original(self, **kwargs)

    monkeypatch.setattr(StageResumeSession, 'flush', flush)
    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_SUCCESS
    assert 'Failed to flush stage resume; will retry' in caplog.text
    assert workspace.read_json(workspace.result('final'))['9'] == 'RU(Unique text 9)'
    report = workspace.read_json('result/sample.json/run_report.json')
    assert report['aggregates']['degraded'] == 0
    assert len(workspace.read_json(workspace.result('resume/translated'))['leaves']) == 12


@pytest.mark.parametrize(('field', 'changed'), [
    ('translation__api__model', 'other-model'),
    ('translation__sampling__style__temperature', 0.321),
    ('processing__required_content_pattern', r'\p{L}{2}'),
    ('source', {'new': 'Changed input'}),
])
def test_continue_refuses_changed_profile_or_source(workspace, model, run_main, caplog, field, changed):
    _interrupt_on(model, 3)
    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED
    _reset_shutdown()
    model.on_model_call = None
    model.requests.clear()
    if field == 'source':
        workspace.write_json('sample.json', changed)
    else:
        workspace.write_configs(**{field: changed})
    assert run_main(*TRANSLATE_NO_CACHE, '--continue') == EXIT_FAILURE_RATIO_EXCEEDED
    assert 'fingerprint does not match' in caplog.text
    assert not model.model_calls


@pytest.mark.parametrize('changed', ['glossary', 'context'])
def test_changed_leaf_context_reprocesses_only_affected_resumed_leaf(workspace, model, run_main, changed):
    _interrupt_on(model, 3)
    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED
    _reset_shutdown()
    model.on_model_call = None
    model.requests.clear()
    if changed == 'glossary':
        workspace.write_json('glossary.json', {'Guild': 'Гильдия'})
    else:
        with (workspace.root / 'context.toml').open('a', encoding='utf-8') as stream:
            stream.write('\n[[leaves]]\npath = \'["ui","title"]\'\nslot = "role"\n'
                         'text = "Use a concise heading."\n')
    assert run_main(*TRANSLATE_NO_CACHE, '--continue') == EXIT_SUCCESS
    assert len(model.model_calls) == 6


def test_forced_interrupt_skips_stage_flush(workspace, model, run_main, monkeypatch):
    def interrupt(_):
        raise RunInterrupted('SIGINT', forced=True)

    def unexpected(*args, **kwargs):
        pytest.fail('Forced interruption must not flush stage progress')

    model.on_model_call = interrupt
    monkeypatch.setattr(StageResumeSession, 'flush_for_shutdown', unexpected)
    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED
    assert not workspace.exists(workspace.result('resume/translated'))


def test_real_sigint_while_journal_lock_is_held_unwinds_before_flush(workspace):
    script = dedent('''
        import inspect
        import signal
        import sys
        import responses
        from llm_conductor.cli.main import main
        from llm_conductor.core.resume import StageResumeSession
        from tests.support import OLLAMA_URL
        from tests.system.harness import EchoModel

        lines, start = inspect.getsourcelines(StageResumeSession.note_accepted)
        target = start + next(i for i, line in enumerate(lines) if 'self._since_flush +=' in line)
        def trace(frame, event, arg):
            if (frame.f_code is StageResumeSession.note_accepted.__code__
                    and event == 'line' and frame.f_lineno == target):
                sys.settrace(None)
                signal.raise_signal(signal.SIGINT)
            return trace

        sys.argv = ['llm-conductor', 'run', '--target', 'sample.json', '--stage', 'translate', '--no-cache']
        with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
            mock.add_callback(responses.POST, OLLAMA_URL, callback=EchoModel())
            sys.settrace(trace)
            main()
    ''')
    env = {**os.environ, 'PYTHONPATH': str(REPO_ROOT),
           'LLMC_CONFIG': str(workspace.root / 'conductor.toml')}
    try:
        result = subprocess.run([sys.executable, '-c', script], cwd=workspace.root,
                                env=env, capture_output=True, text=True, timeout=20, check=False)
    except subprocess.TimeoutExpired:
        pytest.fail('SIGINT deadlocked while the journal lock was held')
    assert result.returncode == EXIT_INTERRUPTED, result.stdout + result.stderr
    index = workspace.read_json(workspace.result('resume/translated'))
    assert len(index['leaves']) == 1
    assert workspace.read_json(workspace.result('translated'))['ui']['title'].startswith('RU(')
